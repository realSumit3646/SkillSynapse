from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from backend.utils.config import settings
from backend.utils.text_utils import dedupe_preserve_order, display_name

# Tested working models (3072 dims each, confirmed via langchain-google-genai v2.1.10)
_EMBEDDING_MODELS = [
    "models/gemini-embedding-001",
    "models/gemini-embedding-2-preview",
    "models/gemini-embedding-2",
]


class EmbeddingClusterService:
    def __init__(self) -> None:
        # Read key from pydantic-settings or fall back to os.environ directly
        self._api_key = (settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")).strip()

        self._embedding_model_idx = 0
        self._embedding_model: GoogleGenerativeAIEmbeddings | None = None

        if self._api_key:
            self._embedding_model = self._make_embedding_client(self._embedding_model_idx)
        else:
            print("WARNING [embeddings]: No GEMINI_API_KEY — falling back to TF-IDF for skill clustering.")

        self.chat_models = self._build_chat_models(settings.gemini_chat_model)
        self.chat_model_idx = 0
        self.naming_chain: RunnableSequence | None = None
        if settings.allow_llm_cluster_naming and self._api_key and self.chat_models:
            self.naming_prompt = PromptTemplate.from_template(
                """
Pick one representative skill from this list.
Rules:
- Return exactly one item.
- Must be one of the given skills, verbatim.
- No explanation.

Skills:
{skills}
""".strip()
            )
            self.naming_chain = self._build_naming_chain(self.chat_models[self.chat_model_idx])
        else:
            self.naming_prompt = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def group_and_reduce(self, skills: Iterable[str], max_skills: int = 10) -> list[str]:
        unique_skills = dedupe_preserve_order(skills)
        if not unique_skills:
            return []

        if len(unique_skills) <= max_skills:
            return [display_name(s) for s in unique_skills]

        vectors = self._embed(unique_skills)
        labels = self._cluster(vectors)
        clusters = self._labels_to_clusters(unique_skills, vectors, labels)

        while len(clusters) > max_skills:
            clusters = self._merge_closest_clusters(clusters)

        parents = []
        for cluster in clusters:
            parent = self._select_parent(cluster["skills"], cluster["vectors"])
            parents.append(display_name(parent))

        parents = dedupe_preserve_order(parents)
        return parents[:max_skills]

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, skills: list[str]) -> np.ndarray:
        if self._embedding_model is not None:
            while True:
                try:
                    vectors = self._embedding_model.embed_documents(skills)
                    return np.asarray(vectors, dtype=np.float32)
                except Exception as exc:
                    next_idx = self._embedding_model_idx + 1
                    if next_idx < len(_EMBEDDING_MODELS):
                        print(
                            f"WARNING [embeddings]: {_EMBEDDING_MODELS[self._embedding_model_idx]} failed "
                            f"({type(exc).__name__}), trying {_EMBEDDING_MODELS[next_idx]}"
                        )
                        self._embedding_model_idx = next_idx
                        self._embedding_model = self._make_embedding_client(next_idx)
                        continue
                    print(f"WARNING [embeddings]: all Gemini models exhausted — falling back to TF-IDF. Error: {exc}")
                    self._embedding_model = None
                    break

        return self._tfidf_embed(skills)

    def _make_embedding_client(self, idx: int) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(
            google_api_key=self._api_key,
            model=_EMBEDDING_MODELS[idx],
        )

    @staticmethod
    def _tfidf_embed(skills: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        matrix = vec.fit_transform(skills).toarray().astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster(self, vectors: np.ndarray) -> np.ndarray:
        if len(vectors) == 1:
            return np.array([0])

        similarity = cosine_similarity(vectors)
        distance = 1.0 - similarity
        model = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=settings.cluster_distance_threshold,
            n_clusters=None,
        )
        return model.fit_predict(distance)

    @staticmethod
    def _labels_to_clusters(skills: list[str], vectors: np.ndarray, labels: np.ndarray) -> list[dict]:
        grouped: dict[int, dict] = defaultdict(lambda: {"skills": [], "vectors": []})
        for idx, label in enumerate(labels):
            grouped[int(label)]["skills"].append(skills[idx])
            grouped[int(label)]["vectors"].append(vectors[idx])

        clusters = []
        for _, item in grouped.items():
            cluster_vectors = np.asarray(item["vectors"], dtype=np.float32)
            clusters.append({
                "skills": item["skills"],
                "vectors": cluster_vectors,
                "centroid": cluster_vectors.mean(axis=0),
            })
        return clusters

    @staticmethod
    def _merge_closest_clusters(clusters: list[dict]) -> list[dict]:
        centroids = np.asarray([c["centroid"] for c in clusters], dtype=np.float32)
        sim = cosine_similarity(centroids)
        np.fill_diagonal(sim, -1.0)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)

        merged_vectors = np.vstack([clusters[i]["vectors"], clusters[j]["vectors"]])
        new_clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)]
        new_clusters.append({
            "skills": clusters[i]["skills"] + clusters[j]["skills"],
            "vectors": merged_vectors,
            "centroid": merged_vectors.mean(axis=0),
        })
        return new_clusters

    def _select_parent(self, skills: list[str], vectors: np.ndarray) -> str:
        if len(skills) == 1:
            return skills[0]
        if self.naming_chain:
            choice = self._safe_llm_parent(skills)
            if choice:
                return choice
        centroid = vectors.mean(axis=0, keepdims=True)
        sim = cosine_similarity(vectors, centroid).flatten()
        return skills[int(np.argmax(sim))]

    # ------------------------------------------------------------------
    # LLM naming chain
    # ------------------------------------------------------------------

    async def select_parent_async(self, skills: list[str]) -> str:
        if len(skills) == 1:
            return skills[0]
        if self.naming_chain:
            while True:
                try:
                    raw = await self.naming_chain.ainvoke({"skills": "\n".join(f"- {s}" for s in skills)})
                    candidate = raw.strip()
                    return candidate if candidate in skills else skills[0]
                except Exception as exc:
                    if self._is_rate_limit_error(exc) and self.chat_model_idx + 1 < len(self.chat_models):
                        self.chat_model_idx += 1
                        self.naming_chain = self._build_naming_chain(self.chat_models[self.chat_model_idx])
                        continue
                    return skills[0]
        return skills[0]

    def _safe_llm_parent(self, skills: list[str]) -> str | None:
        if not self.naming_chain:
            return None
        while True:
            try:
                candidate = self.naming_chain.invoke({"skills": "\n".join(f"- {s}" for s in skills)}).strip()
                return candidate if candidate in skills else None
            except Exception as exc:
                if self._is_rate_limit_error(exc) and self.chat_model_idx + 1 < len(self.chat_models):
                    self.chat_model_idx += 1
                    self.naming_chain = self._build_naming_chain(self.chat_models[self.chat_model_idx])
                    continue
                return None

    def _build_naming_chain(self, model_name: str) -> RunnableSequence:
        llm = ChatGoogleGenerativeAI(
            google_api_key=self._api_key,
            model=model_name.removeprefix("models/"),
            temperature=0,
        )
        return self.naming_prompt | llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        err = str(exc).lower()
        return "429" in err or "resource_exhausted" in err or "rate" in err

    @staticmethod
    def _build_chat_models(primary: str) -> list[str]:
        candidates = [
            primary,
            "models/gemini-2.5-flash",
            "models/gemini-2.5-flash-lite",
            "models/gemini-flash-latest",
            "models/gemini-flash-lite-latest",
            "models/gemini-2.5-pro",
            "models/gemini-pro-latest",
            "models/gemini-3-flash-preview",
            "models/gemini-3.1-flash-lite-preview",
        ]
        seen: set[str] = set()
        result: list[str] = []
        for m in candidates:
            k = m.strip()
            if k and k not in seen:
                seen.add(k)
                result.append(k)
        return result
