"""Priority Scoring Algorithm for Skills."""

import re
from typing import Dict, List, Tuple
from backend.learning_path.models import SkillMetadata, SkillWithScore, ScoringConfig


class ScoringService:
    """Calculates priority scores for skills based on difficulty, time, and impact."""
    
    def __init__(self, config: ScoringConfig = None):
        """Initialize with optional custom configuration."""
        self.config = config or ScoringConfig()
    
    def parse_time_to_days(self, time_str: str) -> float:
        """
        Convert time string to approximate days.
        
        Examples:
            "3 days" → 3
            "1 week" → 7
            "2 weeks" → 14
            "2-3 months" → 60 (uses midpoint)
            "3-4 weeks" → 24.5
        """
        time_str = time_str.lower().strip()
        
        # Handle ranges (e.g., "2-3 months")
        range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*(day|week|month)', time_str)
        if range_match:
            start, end, unit = int(range_match.group(1)), int(range_match.group(2)), range_match.group(3)
            midpoint = (start + end) / 2
            return self._unit_to_days(midpoint, unit)
        
        # Handle single values (e.g., "3 days")
        single_match = re.search(r'(\d+\.?\d*)\s*(day|week|month)', time_str)
        if single_match:
            value, unit = float(single_match.group(1)), single_match.group(2)
            return self._unit_to_days(value, unit)
        
        # Default to 7 days if parsing fails
        return 7.0
    
    @staticmethod
    def _unit_to_days(value: float, unit: str) -> float:
        """Convert time unit to days."""
        conversions = {
            'day': 1,
            'week': 7,
            'month': 30,
        }
        return value * conversions.get(unit, 1)
    
    def calculate_score(self, skill: SkillMetadata) -> float:
        """
        Calculate priority score for a skill.
        
        Formula: Score = (unlock_power * time_weight) / (difficulty^difficulty_exponent * time_weight)
        
        Logic:
            - Higher unlock_power → Higher score (more valuable skill)
            - Higher difficulty → Lower score (harder to prioritize)
            - Longer time → Lower score (take shorter skills first)
        
        Args:
            skill: SkillMetadata object with difficulty, time, unlock_power
            
        Returns:
            float: Priority score (higher = learn first)
        """
        time_days = self.parse_time_to_days(skill.time)
        
        # Normalize time to weeks for better scaling
        time_weeks = max(time_days / 7, 0.1)  # Avoid division by zero
        
        # Apply formula with configurable weights
        numerator = (
            skill.unlock_power * self.config.unlock_power_weight
        )
        
        denominator = (
            (skill.difficulty ** self.config.difficulty_exponent) * 
            (time_weeks * self.config.time_weight)
        )
        
        score = numerator / denominator if denominator > 0 else 0
        return round(score, 2)
    
    def score_all_skills(self, skills_metadata: Dict[str, dict]) -> List[SkillWithScore]:
        """
        Score all skills from metadata dictionary.
        
        Args:
            skills_metadata: {skill_name: {difficulty, time, unlock_power, ...}}
            
        Returns:
            List of SkillWithScore sorted by score (highest first)
        """
        scored_skills = []
        
        for skill_name, attributes in skills_metadata.items():
            try:
                # Create SkillMetadata instance
                skill = SkillMetadata(
                    name=skill_name,
                    difficulty=attributes.get('difficulty', 5),
                    time=attributes.get('time', '1 week'),
                    unlock_power=attributes.get('unlock_power', 5),
                    prerequisites=attributes.get('prerequisites', []),
                    description=attributes.get('description'),
                )
                
                # Calculate score
                score = self.calculate_score(skill)
                
                # Create scored skill
                scored_skill = SkillWithScore(
                    name=skill.name,
                    difficulty=skill.difficulty,
                    time=skill.time,
                    unlock_power=skill.unlock_power,
                    score=score,
                    prerequisites=skill.prerequisites,
                    description=skill.description,
                )
                scored_skills.append(scored_skill)
                
            except Exception as e:
                print(f"Warning: Skipping skill '{skill_name}': {e}")
                continue
        
        # Sort by score (highest first)
        scored_skills.sort(key=lambda x: x.score, reverse=True)
        return scored_skills
    
    def get_score_explanation(self, skill: SkillMetadata) -> str:
        """
        Get human-readable explanation of the score calculation.
        
        Useful for debugging and understanding prioritization.
        """
        score = self.calculate_score(skill)
        time_days = self.parse_time_to_days(skill.time)
        time_weeks = time_days / 7
        
        return (
            f"Skill: {skill.name}\n"
            f"  Unlock Power: {skill.unlock_power}\n"
            f"  Difficulty: {skill.difficulty}/10\n"
            f"  Time: {skill.time} (~{time_weeks:.1f} weeks)\n"
            f"  Score: {score:.2f}\n"
            f"  Reasoning: Higher unlock power and shorter time → higher priority. "
            f"Higher difficulty → lower priority."
        )
