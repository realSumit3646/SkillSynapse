from typing import Any

from pydantic import BaseModel


class SkillMetrics(BaseModel):
    score: float
    confidence: float
    difficulty: int
    time: int
    unlock_power: int


class AnalyzeSkillsResponse(BaseModel):
    needs_feedback: list[str]
    all_skills: dict[str, SkillMetrics]
    skill_gaps: dict[str, dict[str, Any]]
