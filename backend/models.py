from pydantic import BaseModel, Field


class SkillExtractionRequest(BaseModel):
    job_description: str = Field(..., min_length=1)


class SkillExtractionResponse(BaseModel):
    skills: list[str]
