from __future__ import annotations

import anyio
from fastapi import APIRouter, HTTPException, status

from backend.models import SkillExtractionRequest, SkillExtractionResponse
from backend.services.embedding_cluster import EmbeddingClusterService
from backend.services.skill_extractor import SkillExtractorService
from backend.utils.config import settings

router = APIRouter(tags=["skills"])

skill_extractor = SkillExtractorService()
cluster_service = EmbeddingClusterService()


@router.post("/extract-skills", response_model=SkillExtractionResponse)
async def extract_skills(payload: SkillExtractionRequest) -> SkillExtractionResponse:
    source_text = payload.job_description

    if not source_text.strip():
        return SkillExtractionResponse(skills=[])

    try:
        extracted = await skill_extractor.extract_skills(source_text)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    reduced = await anyio.to_thread.run_sync(
        cluster_service.group_and_reduce,
        extracted,
        settings.max_output_skills,
    )
    return SkillExtractionResponse(skills=reduced)
