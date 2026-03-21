from __future__ import annotations

import anyio
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.services.embedding_cluster import EmbeddingClusterService
from backend.services.skill_extractor import SkillExtractorService
from backend.skill_proficiency.models import AnalyzeSkillsResponse
from backend.skill_proficiency.services.analysis_service import (
    build_similarity_maps,
    detect_skills_with_evidence,
    finalize_metrics,
)
from backend.skill_proficiency.services.parsing_service import extract_resume_text, parse_feedback
from backend.skill_proficiency.utils.common import normalize_text
from backend.utils.config import settings

router = APIRouter(tags=["skills"])

skill_extractor = SkillExtractorService()
cluster_service = EmbeddingClusterService()


class AnalysisState:
    """Store intermediate analysis data for reuse across routes."""

    job_description: Optional[str] = None
    resume_text: Optional[str] = None
    required_skills: Optional[list[str]] = None
    evidence: Optional[dict[str, dict[str, float]]] = None
    sim_map: Optional[dict[str, float]] = None
    closest_map: Optional[dict[str, str | None]] = None


analysis_state = AnalysisState()


@router.post("/analyze-skills", response_model=AnalyzeSkillsResponse)
async def analyze_skills_from_job_description(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    user_feedback: str | None = Form(default=None),
) -> AnalyzeSkillsResponse:
    source_text = (job_description or "").strip()
    if not source_text:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="job_description is required.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    feedback = parse_feedback(user_feedback)

    raw_resume_text = extract_resume_text(file, file_bytes)
    normalized_resume_text = normalize_text(raw_resume_text)
    if not normalized_resume_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No readable text found in resume.")

    try:
        extracted = await skill_extractor.extract_skills(source_text)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    required_skills = await anyio.to_thread.run_sync(
        cluster_service.group_and_reduce,
        extracted,
        settings.max_output_skills,
    )

    if not required_skills:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No skills could be extracted from job_description.",
        )

    evidence = detect_skills_with_evidence(normalized_resume_text, required_skills)
    detected_skills = [s for s, v in evidence.items() if v.get("mentions", 0.0) > 0.0]
    sim_map, closest_map = build_similarity_maps(required_skills, detected_skills)

    # Store state for potential reuse with different feedback
    analysis_state.job_description = source_text
    analysis_state.resume_text = normalized_resume_text
    analysis_state.required_skills = required_skills
    analysis_state.evidence = evidence
    analysis_state.sim_map = sim_map
    analysis_state.closest_map = closest_map

    return finalize_metrics(required_skills, evidence, feedback, sim_map, closest_map)


@router.post("/provide-feedback", response_model=AnalyzeSkillsResponse)
async def provide_feedback_and_reanalyze(user_feedback: str = Form(default=None)) -> AnalyzeSkillsResponse:
    """Use stored analysis state with new feedback to recompute scores.

    Call /analyze-skills first to store the job description and resume.
    Then call this route with different feedback to get updated results.
    """
    if not analysis_state.resume_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No stored resume. Please call /analyze-skills first.",
        )

    feedback = parse_feedback(user_feedback)
    required_skills = analysis_state.required_skills or []
    evidence = analysis_state.evidence or {}
    sim_map = analysis_state.sim_map or {}
    closest_map = analysis_state.closest_map or {}

    return finalize_metrics(required_skills, evidence, feedback, sim_map, closest_map)