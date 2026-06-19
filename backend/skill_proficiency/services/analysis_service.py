from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import numpy as np
from flashtext import KeywordProcessor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.skill_proficiency.models import AnalyzeSkillsResponse, SkillMetrics
from backend.skill_proficiency.utils.common import (
    YEAR_PATTERN,
    clamp,
    classify_base_time_days,
    compute_unlock_power,
)
from backend.utils.config import settings

GEMINI_RUBRIC_INDICATORS: tuple[str, ...] = (
    "explicit_mention",
    "project_usage",
    "real_world_application",
    "multiple_usage",
    "recent_usage",
    "impact_evidence",
    "tool_ecosystem",
    "problem_solving",
    "depth_indicator",
    "consistency",
)


class GeminiEvidenceScorer:
    def __init__(self) -> None:
        self.chat_models = self._build_chat_models(settings.gemini_chat_model)
        self.chat_model_idx = 0
        self.chain: RunnableSequence | None = None

        self.prompt = PromptTemplate.from_template(
            """
Task: Evaluate resume evidence for each required skill using the rubric indicators below.

Rubric indicators (binary 0 or 1):
- explicit_mention: Skill appears explicitly.
- project_usage: Used in projects/capstone/builds.
- real_world_application: Used in internship/work/client/production settings.
- multiple_usage: Mentioned in multiple distinct contexts.
- recent_usage: Evidence that usage is recent.
- impact_evidence: Evidence of measurable impact/outcomes.
- tool_ecosystem: Related tools/frameworks around the skill are used.
- problem_solving: Skill linked to solving/debugging/optimization work.
- depth_indicator: Evidence of advanced/deep usage.
- consistency: Skill appears consistently across resume sections.

Return ONLY valid JSON object, no markdown, no explanation.

Expected format:
{{
  "skill_name": {{
    "explicit_mention": 0,
    "project_usage": 0,
    "real_world_application": 0,
    "multiple_usage": 0,
    "recent_usage": 0,
    "impact_evidence": 0,
    "tool_ecosystem": 0,
    "problem_solving": 0,
    "depth_indicator": 0,
    "consistency": 0,
    "confidence": 0.0
  }}
}}

Rules:
- Use only evidence present in the resume text.
- Keys must be lowercase and exactly match the provided required skills.
- confidence must be between 0.0 and 1.0.

Required skills (lowercase JSON array):
{skills_json}

Resume text:
{text}
""".strip()
        )

        if settings.gemini_api_key and self.chat_models:
            self.chain = self._build_chain(self.chat_models[self.chat_model_idx])

    def score_skills(self, text: str, required_skills: list[str]) -> dict[str, dict[str, float]]:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini-based proficiency scoring")
        if self.chain is None:
            raise RuntimeError("Gemini scoring chain is not available")

        payload = {
            "skills_json": json.dumps([s.lower() for s in required_skills]),
            "text": text,
        }

        raw = ""
        while True:
            try:
                raw = self.chain.invoke(payload)
                break
            except Exception as exc:
                if self._is_rate_limit_error(exc) and self.chat_model_idx + 1 < len(self.chat_models):
                    self.chat_model_idx += 1
                    self.chain = self._build_chain(self.chat_models[self.chat_model_idx])
                    continue
                raise

        parsed = self._parse_json_object(raw)
        if parsed is None:
            parsed = self._parse_first_json_object(raw)
        if parsed is None:
            raise RuntimeError("Failed to parse Gemini scoring output as JSON object")

        normalized: dict[str, dict[str, float]] = {}
        for skill in required_skills:
            key = skill.lower()
            item = parsed.get(key, {}) if isinstance(parsed, dict) else {}
            if not isinstance(item, dict):
                item = {}

            normalized[key] = {
                indicator: float(1 if _to_float(item.get(indicator, 0.0)) >= 0.5 else 0)
                for indicator in GEMINI_RUBRIC_INDICATORS
            }
            normalized[key]["confidence"] = clamp(_to_float(item.get("confidence", 0.0)), 0.0, 1.0)

        return normalized

    def _build_chain(self, model_name: str) -> RunnableSequence:
        llm = ChatGoogleGenerativeAI(
            google_api_key=settings.gemini_api_key,
            model=model_name.removeprefix("models/"),
            temperature=0,
        )
        return self.prompt | llm | StrOutputParser()

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        err = str(exc).lower()
        return "429" in err or "resource_exhausted" in err or "rate" in err

    @staticmethod
    def _parse_json_object(value: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _parse_first_json_object(value: str) -> dict[str, Any] | None:
        match = re.search(r"\{[\s\S]*\}", value)
        if not match:
            return None
        return GeminiEvidenceScorer._parse_json_object(match.group(0))

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

        models: list[str] = []
        seen: set[str] = set()
        for model in candidates:
            key = model.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            models.append(key)
        return models


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


GEMINI_EVIDENCE_SCORER = GeminiEvidenceScorer()


def build_keyword_processor(required_skills: list[str]) -> KeywordProcessor:
    kp = KeywordProcessor(case_sensitive=False)
    for skill in required_skills:
        kp.add_keyword(skill, skill.lower())
    return kp


def get_context_window(text: str, start: int, end: int, radius: int = 80) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right]


def detect_skills_with_evidence(text: str, required_skills: list[str]) -> dict[str, dict[str, float]]:
    normalized_text = text.lower()
    kp = build_keyword_processor(required_skills)
    matches = kp.extract_keywords(normalized_text, span_info=True)

    gemini_scores = GEMINI_EVIDENCE_SCORER.score_skills(text=text, required_skills=required_skills)

    evidence: dict[str, dict[str, Any]] = {
        skill.lower(): {
            "mentions": 0,
            "latest_year": None,
        }
        for skill in required_skills
    }

    for found_skill, start, end in matches:
        item = evidence[found_skill]
        item["mentions"] += 1

        window = get_context_window(normalized_text, start, end)

        years = [int(y) for y in YEAR_PATTERN.findall(window)]
        if years:
            latest = max(years)
            if item["latest_year"] is None or latest > item["latest_year"]:
                item["latest_year"] = latest

    scored: dict[str, dict[str, float]] = {}
    current_year = datetime.now().year

    for skill, item in evidence.items():
        mentions = int(item["mentions"])
        latest_year = item["latest_year"]
        llm_item = gemini_scores.get(skill, {})

        explicit_mention = float(llm_item.get("explicit_mention", 1.0 if mentions > 0 else 0.0))
        project_usage = float(llm_item.get("project_usage", 0.0))
        real_world_application = float(llm_item.get("real_world_application", 0.0))
        multiple_usage = float(llm_item.get("multiple_usage", 1.0 if mentions >= 2 else 0.0))
        recent_usage_by_year = 1.0 if latest_year is not None and latest_year >= current_year - 2 else 0.0
        recent_usage = max(float(llm_item.get("recent_usage", 0.0)), recent_usage_by_year)
        impact_evidence = float(llm_item.get("impact_evidence", 0.0))
        tool_ecosystem = float(llm_item.get("tool_ecosystem", 0.0))
        problem_solving = float(llm_item.get("problem_solving", 0.0))
        depth_indicator = float(llm_item.get("depth_indicator", 0.0))
        consistency = float(llm_item.get("consistency", 0.0))

        score_model = (
            explicit_mention
            + project_usage
            + real_world_application
            + multiple_usage
            + recent_usage
            + impact_evidence
            + tool_ecosystem
            + problem_solving
            + depth_indicator
            + consistency
        )

        mentions_norm = min(mentions / 4.0, 1.0)
        indicator_density = score_model / 10.0
        llm_confidence = clamp(float(llm_item.get("confidence", 0.0)), 0.0, 1.0)
        confidence = clamp(0.6 * llm_confidence + 0.25 * indicator_density + 0.15 * mentions_norm, 0.0, 1.0)

        scored[skill] = {
            "mentions": float(mentions),
            "recent_usage": float(recent_usage),
            "consistency": float(consistency),
            "score_model": float(score_model),
            "confidence": float(confidence),
        }

    return scored


def build_similarity_maps(required_skills: list[str], detected_skills: list[str]) -> tuple[dict[str, float], dict[str, str | None]]:
    if not detected_skills:
        return {s.lower(): 0.0 for s in required_skills}, {s.lower(): None for s in required_skills}

    all_texts = required_skills + detected_skills
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    vectorizer.fit(all_texts)

    req_vectors = vectorizer.transform(required_skills)
    det_vectors = vectorizer.transform(detected_skills)
    sim_matrix = cosine_similarity(req_vectors, det_vectors)

    sim_map: dict[str, float] = {}
    closest_map: dict[str, str | None] = {}
    for i, req in enumerate(required_skills):
        row = sim_matrix[i]
        j = int(np.argmax(row))
        sim_map[req.lower()] = float(row[j])
        closest_map[req.lower()] = detected_skills[j].lower()
    return sim_map, closest_map


def finalize_metrics(
    required_skills: list[str],
    evidence: dict[str, dict[str, float]],
    user_feedback: dict[str, float],
    sim_map: dict[str, float],
    closest_map: dict[str, str | None],
) -> AnalyzeSkillsResponse:
    all_skills: dict[str, SkillMetrics] = {}
    needs_feedback: list[str] = []

    has_feedback = bool(user_feedback)

    for skill in required_skills:
        key = skill.lower()
        item = evidence.get(
            key,
            {
                "score_model": 0.0,
                "confidence": 0.0,
                "mentions": 0.0,
                "recent_usage": 0.0,
                "consistency": 0.0,
            },
        )

        score_model = float(item["score_model"])
        confidence = float(item["confidence"])
        mentions = float(item.get("mentions", 0.0))
        recent_usage = float(item.get("recent_usage", 0.0))
        consistency = float(item.get("consistency", 0.0))
        sim = clamp(sim_map.get(key, 0.0), 0.0, 1.0)

        if key in user_feedback:
            score = 0.6 * score_model + 0.4 * user_feedback[key]
            confidence = 0.9
        else:
            # Robust no-feedback score: blend rubric evidence with mention strength and semantic support.
            rubric_norm = clamp(score_model / 10.0, 0.0, 1.0)
            mention_strength = min(mentions / 3.0, 1.0)
            explicit_mention = 1.0 if mentions > 0.0 else 0.0

            # If skill is not explicitly mentioned, semantic similarity should not dominate.
            semantic_support = sim if explicit_mention > 0 else min(sim, 0.55)

            evidence_strength = (
                0.55 * rubric_norm
                + 0.25 * mention_strength
                + 0.10 * consistency
                + 0.10 * recent_usage
            )

            score = 10.0 * (0.80 * evidence_strength + 0.20 * semantic_support)

            # Prevent high scores for skills inferred only via semantic proximity.
            if explicit_mention == 0.0:
                score = min(score, 4.0)

            # Stabilize strong evidence cases against underestimation.
            if mentions >= 3.0 and score_model >= 6.0:
                score = max(score, 6.0)

            # Recalibrate confidence in no-feedback mode using robustness signals.
            confidence = clamp(
                0.45 * confidence
                + 0.25 * mention_strength
                + 0.20 * explicit_mention
                + 0.10 * consistency,
                0.0,
                1.0,
            )

            if explicit_mention == 0.0:
                confidence = min(confidence, 0.45)

            if confidence < 0.65 and 2.0 <= score <= 7.0:
                needs_feedback.append(skill)

        closest_skill = closest_map.get(key)
        related_score = float(evidence.get(closest_skill or "", {}).get("score_model", 0.0))

        difficulty_raw = 10.0 * (1.0 - sim) - 0.3 * related_score
        difficulty = int(round(clamp(difficulty_raw, 1.0, 10.0)))

        base_time = classify_base_time_days(skill)
        time_days = base_time * (1.0 - sim) + base_time * 0.3
        time_days_int = max(1, int(round(time_days)))

        unlock_power = compute_unlock_power(skill)

        metric = SkillMetrics(
            score=round(clamp(score, 0.0, 10.0), 2),
            confidence=round(clamp(confidence, 0.0, 1.0), 3),
            difficulty=difficulty,
            time=time_days_int,
            unlock_power=unlock_power,
        )
        all_skills[skill] = metric

    skill_gaps: dict[str, dict[str, Any]] = {}
    avg_score = sum(metric.score for metric in all_skills.values()) / len(all_skills) if all_skills else 0.0

    for skill, metric in all_skills.items():
        if metric.score < avg_score:
            skill_gaps[skill] = {
                "difficulty": metric.difficulty,
                "time": metric.time,
                "unlock_power": metric.unlock_power,
            }

    return AnalyzeSkillsResponse(
        needs_feedback=needs_feedback if not has_feedback else [],
        all_skills=all_skills,
        skill_gaps=skill_gaps,
    )
