"""Configuration for Learning Path Generation."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class LearningPathSettings(BaseSettings):
    """Settings for learning path generation algorithm."""
    
    model_config = SettingsConfigDict(
        env_file=("backend/.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Scoring Algorithm Configuration
    difficulty_exponent: float = 1.5
    """Exponent for difficulty penalty in scoring formula. Higher = more penalty for hard skills."""
    
    time_weight: float = 1.0
    """Weight multiplier for time factor. Higher = more penalty for time-consuming skills."""
    
    unlock_power_weight: float = 1.0
    """Weight multiplier for unlock power (skill value). Higher = more reward for valuable skills."""
    
    # Learning Path Generation
    max_output_skills: int = 15
    """Maximum skills to include in learning path."""
    
    max_difficulty_per_phase: int = 8
    """Maximum difficulty for skills to learn in parallel (1-10)."""
    
    # Future Extensibility Flags (for v2 features)
    enable_learning_curve_adjustment: bool = False
    """(V2) Adjust time estimates based on learning curve data."""
    
    enable_job_market_weighting: bool = False
    """(V2) Weight skills by current job market demand."""
    
    enable_spaced_repetition: bool = False
    """(V2) Include spaced repetition scheduling."""
    
    enable_motivation_balancing: bool = False
    """(V2) Balance difficulty to prevent burnout."""
    
    enable_parallel_path_optimization: bool = False
    """(V2) Optimize track assignment for max parallelization."""


# Create global settings instance
learning_path_settings = LearningPathSettings()
