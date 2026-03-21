from pydantic import BaseModel, Field
from typing import Optional, List


class SkillMetadata(BaseModel):
    """Metadata for a single skill with learning constraints."""
    name: str = Field(..., description="Skill name")
    difficulty: int = Field(..., ge=1, le=10, description="Difficulty level (1-10)")
    time: str = Field(..., description="Estimated time to learn (e.g., '3 days', '2 weeks', '2-3 months')")
    unlock_power: int = Field(..., ge=1, description="Impact/value of this skill")
    prerequisites: List[str] = Field(default_factory=list, description="List of skill names that are prerequisites")
    description: Optional[str] = Field(None, description="Optional description of the skill")


class SkillWithScore(BaseModel):
    """Skill with calculated priority score."""
    name: str
    difficulty: int
    time: str
    unlock_power: int
    score: float = Field(..., description="Calculated priority score")
    prerequisites: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class LearningPathRequest(BaseModel):
    """Request to generate a learning path."""
    skills_to_learn: List[str] = Field(..., description="Skills that need to be learned")
    skill_metadata: dict = Field(
        ..., 
        description="Metadata for each skill: {skill_name: {difficulty, time, unlock_power, prerequisites?, description?}}"
    )
    auto_extract_prerequisites: bool = Field(True, description="Use LLM to extract prerequisites if not provided")
    available_time_weeks: Optional[int] = Field(None, description="Total weeks available for learning")
    max_difficulty_per_phase: Optional[int] = Field(8, description="Max difficulty for parallel learning")


class LearningPathResponse(BaseModel):
    """Generated learning roadmap with DAG structure."""
    skills_to_learn: List[str] = Field(..., description="All skills to be learned")
    scored_skills: List[SkillWithScore] = Field(..., description="All skills with calculated priority scores")
    prerequisites_map: dict = Field(..., description="Prerequisite mapping {skill: [prereq_skills]}")
    dag: dict = Field(..., description="Directed Acyclic Graph {skill: {prerequisites: [...], dependents: [...]}}")
    graph: dict = Field(..., description="Lightweight graph for visualization {nodes: [...], edges: [...]}")
    learning_sequence: List[str] = Field(..., description="Recommended learning order (topological sort by priority)")
    tracks: dict = Field(..., description="Parallel learning tracks {track_name: [skills]}")
    metadata: dict = Field(default_factory=dict, description="Additional planning metadata")


class GraphResponse(BaseModel):
    """Lightweight graph structure for visualization."""
    nodes: List[dict] = Field(..., description="Graph nodes with id, difficulty, score, etc.")
    edges: List[dict] = Field(..., description="Graph edges with from, to, weight, type")


class ScoringConfig(BaseModel):
    """Configuration for priority scoring algorithm."""
    # Formula: Score = (unlock_power * time_weight) / (difficulty^difficulty_exponent)
    difficulty_exponent: float = Field(1.5, description="Exponent for difficulty penalty")
    time_weight: float = Field(1.0, description="Weight multiplier for time factor")
    unlock_power_weight: float = Field(1.0, description="Weight multiplier for unlock power")
    
    # Future extensibility
    enable_learning_curve_adjustment: bool = Field(False, description="Adjust scores based on learning curve data")
    enable_job_market_weighting: bool = Field(False, description="Weight skills by job market demand")
