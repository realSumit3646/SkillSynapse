from __future__ import annotations

from typing import Any

import numpy as np
from flashtext import KeywordProcessor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from models import AnalyzeSkillsResponse, SkillMetrics
from utils.common import (
    NEAR_STRONG,
    NEAR_WEAK,
    YEAR_PATTERN,
    clamp,
    classify_base_time_days,
    compute_unlock_power,
)

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


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
    kp = build_keyword_processor(required_skills)
    matches = kp.extract_keywords(text, span_info=True)

    evidence: dict[str, dict[str, Any]] = {
        skill.lower(): {
            "mentions": 0,
            "context_score": 0,
            "latest_year": None,
        }
        for skill in required_skills
    }

    for found_skill, start, end in matches:
        item = evidence[found_skill]
        item["mentions"] += 1

        window = get_context_window(text, start, end)
        if any(token in window for token in NEAR_STRONG):
            item["context_score"] += 2
        elif any(token in window for token in NEAR_WEAK):
            item["context_score"] += 1

        years = [int(y) for y in YEAR_PATTERN.findall(window)]
        if years:
            latest = max(years)
            if item["latest_year"] is None or latest > item["latest_year"]:
                item["latest_year"] = latest

    scored: dict[str, dict[str, float]] = {}
    for skill, item in evidence.items():
        latest_year = item["latest_year"]
        if latest_year is None:
            recency = 0
        elif latest_year >= 2024:
            recency = 2
        elif latest_year >= 2022:
            recency = 1
        else:
            recency = 0

        mentions = int(item["mentions"])
        context_score = int(item["context_score"])
        m = min(mentions, 5)

        score_model = min(1.5 * m + context_score + recency, 10.0)

        mentions_norm = min(mentions / 5.0, 1.0)
        context_norm = min(context_score / 3.0, 1.0)
        recency_norm = recency / 2.0
        confidence = clamp(0.4 * mentions_norm + 0.3 * context_norm + 0.3 * recency_norm, 0.0, 1.0)

        scored[skill] = {
            "mentions": float(mentions),
            "context_score": float(context_score),
            "recency_score": float(recency),
            "score_model": float(score_model),
            "confidence": float(confidence),
        }

    return scored


def build_similarity_maps(required_skills: list[str], detected_skills: list[str]) -> tuple[dict[str, float], dict[str, str | None]]:
    req_vectors = EMBEDDING_MODEL.encode(required_skills, convert_to_numpy=True, normalize_embeddings=True)

    if not detected_skills:
        return {s.lower(): 0.0 for s in required_skills}, {s.lower(): None for s in required_skills}

    det_vectors = EMBEDDING_MODEL.encode(detected_skills, convert_to_numpy=True, normalize_embeddings=True)
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
    skill_gaps: dict[str, dict[str, Any]] = {}
    needs_feedback: list[str] = []

    has_feedback = bool(user_feedback)

    for skill in required_skills:
        key = skill.lower()
        item = evidence.get(
            key,
            {
                "score_model": 0.0,
                "confidence": 0.0,
            },
        )

        score_model = float(item["score_model"])
        confidence = float(item["confidence"])

        if not has_feedback and confidence < 0.6 and 2.0 <= score_model <= 7.0:
            needs_feedback.append(skill)

        if key in user_feedback:
            score = 0.6 * score_model + 0.4 * user_feedback[key]
            confidence = 0.9
        else:
            score = score_model

        sim = clamp(sim_map.get(key, 0.0), 0.0, 1.0)
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

        if metric.score < 7.0:
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
