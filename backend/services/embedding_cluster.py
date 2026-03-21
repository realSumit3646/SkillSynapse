from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from backend.utils.config import settings
from backend.utils.text_utils import dedupe_preserve_order, display_name


class EmbeddingClusterService:
    def __init__(self) -> None:
        self.use_gemini_embeddings = settings.use_gemini_embeddings
        self.local_embedding_model = SentenceTransformer(settings.sentence_transformer_model)
        self.embedding_model = self.local_embedding_model
        self.embedding_models = self._build_embedding_models(settings.gemini_embedding_model)
        self.embedding_model_idx = 0

        if self.use_gemini_embeddings and settings.gemini_api_key:
            self.embedding_model = self._build_embedding_client(self.embedding_models[self.embedding_model_idx])
        elif self.use_gemini_embeddings and not settings.gemini_api_key:
            self.use_gemini_embeddings = False

        self.chat_models = self._build_chat_models(settings.gemini_chat_model)
        self.chat_model_idx = 0
        self.naming_chain: RunnableSequence | None = None
        if settings.allow_llm_cluster_naming and settings.gemini_api_key and self.chat_models:
            naming_prompt = PromptTemplate.from_template(
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
            self.naming_prompt = naming_prompt
            self.naming_chain = self._build_naming_chain(self.chat_models[self.chat_model_idx])
        else:
            self.naming_prompt = None

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

    def _embed(self, skills: list[str]) -> np.ndarray:
        if self.use_gemini_embeddings:
            while True:
                try:
                    vectors = self.embedding_model.embed_documents(skills)
                    return np.asarray(vectors, dtype=np.float32)
                except Exception as exc:
                    if self._is_rate_limit_error(exc) and self.embedding_model_idx + 1 < len(self.embedding_models):
                        self.embedding_model_idx += 1
                        self.embedding_model = self._build_embedding_client(self.embedding_models[self.embedding_model_idx])
                        continue

                    self.use_gemini_embeddings = False
                    self.embedding_model = self.local_embedding_model
                    break

        vectors = self.local_embedding_model.encode(skills, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

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
            centroid = cluster_vectors.mean(axis=0)
            clusters.append(
                {
                    "skills": item["skills"],
                    "vectors": cluster_vectors,
                    "centroid": centroid,
                }
            )
        return clusters

    @staticmethod
    def _merge_closest_clusters(clusters: list[dict]) -> list[dict]:
        centroids = np.asarray([c["centroid"] for c in clusters], dtype=np.float32)
        sim = cosine_similarity(centroids)
        np.fill_diagonal(sim, -1.0)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)

        merged_skills = clusters[i]["skills"] + clusters[j]["skills"]
        merged_vectors = np.vstack([clusters[i]["vectors"], clusters[j]["vectors"]])
        merged_centroid = merged_vectors.mean(axis=0)

        new_clusters = []
        for idx, c in enumerate(clusters):
            if idx in (i, j):
                continue
            new_clusters.append(c)

        new_clusters.append(
            {
                "skills": merged_skills,
                "vectors": merged_vectors,
                "centroid": merged_centroid,
            }
        )
        return new_clusters

    def _select_parent(self, skills: list[str], vectors: np.ndarray) -> str:
        if len(skills) == 1:
            return skills[0]

        if self.naming_chain:
            llm_choice = self._safe_llm_parent(skills)
            if llm_choice:
                return llm_choice

        centroid = vectors.mean(axis=0, keepdims=True)
        sim = cosine_similarity(vectors, centroid).flatten()
        return skills[int(np.argmax(sim))]

    async def select_parent_async(self, skills: list[str]) -> str:
        if len(skills) == 1:
            return skills[0]
        if self.naming_chain:
            while True:
                try:
                    raw = await self.naming_chain.ainvoke({"skills": "\n".join(f"- {s}" for s in skills)})
                    candidate = raw.strip()
                    if candidate in skills:
                        return candidate
                    return skills[0]
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
                if candidate in skills:
                    return candidate
                return None
            except Exception as exc:
                if self._is_rate_limit_error(exc) and self.chat_model_idx + 1 < len(self.chat_models):
                    self.chat_model_idx += 1
                    self.naming_chain = self._build_naming_chain(self.chat_models[self.chat_model_idx])
                    continue
                return None
        return None

    def _build_naming_chain(self, model_name: str) -> RunnableSequence:
        llm = ChatGoogleGenerativeAI(
            google_api_key=settings.gemini_api_key,
            model=model_name.removeprefix("models/"),
            temperature=0,
        )
        return self.naming_prompt | llm | StrOutputParser()

    @staticmethod
    def _build_embedding_client(model_name: str) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(
            google_api_key=settings.gemini_api_key,
            model=model_name,
        )

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
        return EmbeddingClusterService._dedupe_models(candidates)

    @staticmethod
    def _build_embedding_models(primary: str) -> list[str]:
        candidates = [
            primary,
            "models/gemini-embedding-001",
            "models/gemini-embedding-2-preview",
        ]
        return EmbeddingClusterService._dedupe_models(candidates)

    @staticmethod
    def _dedupe_models(candidates: list[str]) -> list[str]:
        models: list[str] = []
        seen = set()
        for model in candidates:
            key = model.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            models.append(key)
        return models
