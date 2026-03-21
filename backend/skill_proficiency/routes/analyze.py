from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from models import AnalyzeSkillsResponse
from services.analysis_service import build_similarity_maps, detect_skills_with_evidence, finalize_metrics
from services.parsing_service import extract_resume_text, parse_feedback, parse_skills
from utils.common import normalize_text

router = APIRouter()


@router.post("/analyze-skills", response_model=AnalyzeSkillsResponse)
async def analyze_skills(
    file: UploadFile = File(...),
    skills: str = Form(...),
    user_feedback: str | None = Form(default=None),
) -> AnalyzeSkillsResponse:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    required_skills = parse_skills(skills)
    feedback = parse_feedback(user_feedback)

    raw_text = extract_resume_text(file, file_bytes)
    normalized_text = normalize_text(raw_text)
    if not normalized_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No readable text found in resume.")

    evidence = detect_skills_with_evidence(normalized_text, required_skills)
    detected_skills = [s for s, v in evidence.items() if v.get("mentions", 0.0) > 0.0]

    sim_map, closest_map = build_similarity_maps(required_skills, detected_skills)

    return finalize_metrics(required_skills, evidence, feedback, sim_map, closest_map)
