"""Convert skill gap format from /extract-skills to learning path format."""

from typing import Dict, List, Tuple


class SkillGapConverter:
    """Convert between /extract-skills format and learning path format."""
    
    @staticmethod
    def days_to_time_string(days: int) -> str:
        """
        Convert days (integer) to human-readable time string.
        
        Args:
            days: Number of days (integer)
            
        Returns:
            Formatted string like "2 weeks", "3 days", "1 month"
            
        Examples:
            7 → "1 week"
            14 → "2 weeks"
            68 → "9.7 weeks"
            3 → "3 days"
            90 → "13 weeks"
            180 → "25.7 weeks" (or "6 months")
        """
        if days <= 0:
            return "1 day"
        
        # Days: show for days < 7
        if days < 7:
            return f"{days} days"
        
        weeks = days / 7
        
        # Weeks: show for 7 <= days < 90 (7-88 days)
        if days < 90:
            if weeks == int(weeks):
                return f"{int(weeks)} week{'s' if weeks > 1 else ''}"
            else:
                return f"{weeks:.1f} weeks"
        else:
            # Months: show for days >= 90
            months = days / 30
            if months < 12:
                if months == int(months):
                    return f"{int(months)} month{'s' if months > 1 else ''}"
                else:
                    return f"{months:.1f} months"
            else:
                # Years: show for days >= 360
                years = days / 365
                if years == int(years):
                    return f"{int(years)} year{'s' if years > 1 else ''}"
                else:
                    return f"{years:.1f} years"
    
    @staticmethod
    def convert_extract_skills_to_learning_path(
        skill_gaps: Dict[str, dict]
    ) -> Dict[str, dict]:
        """
        Convert /extract-skills format to learning path format.
        
        Input format (from /extract-skills):
        {
            "Python Programming": {
                "difficulty": 4,
                "time": 68,              # Days (integer)
                "unlock_power": 11
            }
        }
        
        Output format (for learning path):
        {
            "Python Programming": {
                "difficulty": 4,
                "time": "9.7 weeks",     # String format
                "unlock_power": 11
            }
        }
        
        Args:
            skill_gaps: Dict from /extract-skills skill_gaps field
            
        Returns:
            Dict in learning path format
        """
        converted = {}
        
        for skill_name, metadata in skill_gaps.items():
            difficulty = metadata.get("difficulty", 5)
            time_days = metadata.get("time", 30)
            unlock_power = metadata.get("unlock_power", 5)
            
            # Convert days to string format
            time_string = SkillGapConverter.days_to_time_string(int(time_days))
            
            converted[skill_name] = {
                "difficulty": int(difficulty),
                "time": time_string,
                "unlock_power": int(unlock_power),
                "prerequisites": metadata.get("prerequisites", [])  # If exists
            }
        
        return converted
    
    @staticmethod
    def create_learning_path_request(
        skill_gaps: Dict[str, dict],
        auto_extract_prerequisites: bool = True,
        available_time_weeks: int = None,
        max_difficulty_per_phase: int = 8
    ) -> Dict:
        """
        Create complete learning path request from skill gaps.
        
        Args:
            skill_gaps: From /extract-skills response skill_gaps field
            auto_extract_prerequisites: Whether to use LLM for prerequisites
            available_time_weeks: Optional time budget
            max_difficulty_per_phase: Max difficulty for parallel learning
            
        Returns:
            Complete LearningPathRequest dict
        """
        # Convert format
        converted_metadata = SkillGapConverter.convert_extract_skills_to_learning_path(skill_gaps)
        
        # Get skill names
        skills_to_learn = list(skill_gaps.keys())
        
        # Build request
        request = {
            "skills_to_learn": skills_to_learn,
            "skill_metadata": converted_metadata,
            "auto_extract_prerequisites": auto_extract_prerequisites,
            "max_difficulty_per_phase": max_difficulty_per_phase
        }
        
        if available_time_weeks:
            request["available_time_weeks"] = available_time_weeks
        
        return request
    
    @staticmethod
    def parse_time_string_to_days(time_string: str) -> float:
        """
        Parse time string back to days (opposite of days_to_time_string).
        
        Args:
            time_string: "2 weeks", "3 days", "1 month", "9.7 weeks"
            
        Returns:
            Number of days
            
        Examples:
            "1 week" → 7
            "2 weeks" → 14
            "3 days" → 3
            "1 month" → 30
            "9.7 weeks" → 68
        """
        time_string = time_string.strip().lower()
        
        # Split into number and unit
        parts = time_string.split()
        
        if len(parts) != 2:
            return 30  # Default to 30 days if parse fails
        
        try:
            value = float(parts[0])
            unit = parts[1]
            
            if unit.startswith("day"):
                return value
            elif unit.startswith("week"):
                return value * 7
            elif unit.startswith("month"):
                return value * 30
            else:
                return 30  # Default
        except (ValueError, IndexError):
            return 30  # Default
    
    @staticmethod
    def get_skill_stats(skill_gaps: Dict[str, dict]) -> Dict:
        """
        Get aggregated statistics from skill gaps.
        
        Args:
            skill_gaps: From /extract-skills
            
        Returns:
            {
                "total_skills": int,
                "total_days": int,
                "total_weeks": float,
                "avg_difficulty": float,
                "hardest_skill": str,
                "longest_skill": str,
                "most_valuable_skill": str
            }
        """
        if not skill_gaps:
            return {}
        
        total_days = 0
        difficulties = []
        unlock_powers = []
        
        hardest_skill = None
        longest_skill = None
        most_valuable_skill = None
        
        hardest_difficulty = 0
        longest_days = 0
        highest_unlock_power = 0
        
        for skill_name, metadata in skill_gaps.items():
            difficulty = metadata.get("difficulty", 5)
            days = metadata.get("time", 30)
            unlock_power = metadata.get("unlock_power", 5)
            
            total_days += days
            difficulties.append(difficulty)
            unlock_powers.append(unlock_power)
            
            # Track extremes
            if difficulty > hardest_difficulty:
                hardest_difficulty = difficulty
                hardest_skill = skill_name
            
            if days > longest_days:
                longest_days = days
                longest_skill = skill_name
            
            if unlock_power > highest_unlock_power:
                highest_unlock_power = unlock_power
                most_valuable_skill = skill_name
        
        return {
            "total_skills": len(skill_gaps),
            "total_days": total_days,
            "total_weeks": round(total_days / 7, 1),
            "avg_difficulty": round(sum(difficulties) / len(difficulties), 1),
            "avg_unlock_power": round(sum(unlock_powers) / len(unlock_powers), 1),
            "hardest_skill": hardest_skill,
            "hardest_difficulty": hardest_difficulty,
            "longest_skill": longest_skill,
            "longest_days": longest_days,
            "most_valuable_skill": most_valuable_skill,
            "highest_unlock_power": highest_unlock_power
        }
