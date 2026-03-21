"""API Routes for Learning Path Generation with DAG."""

from fastapi import APIRouter, HTTPException, status
from backend.learning_path.models import LearningPathRequest, LearningPathResponse, ScoringConfig
from backend.learning_path.scorer import ScoringService
from backend.learning_path.path_generator import LearningPathGenerator
from backend.learning_path.dag_builder import DAGBuilder
from backend.learning_path.prerequisite_extractor import PrerequisiteExtractor
from backend.learning_path.config import learning_path_settings
from backend.learning_path.skill_gap_converter import SkillGapConverter

router = APIRouter(tags=["learning-path"], prefix="/learning-path")

# Initialize services
scorer = ScoringService(config=ScoringConfig(
    difficulty_exponent=learning_path_settings.difficulty_exponent,
    time_weight=learning_path_settings.time_weight,
    unlock_power_weight=learning_path_settings.unlock_power_weight,
))
dag_builder = DAGBuilder()
prerequisite_extractor = PrerequisiteExtractor()
path_generator = LearningPathGenerator(
    scorer=scorer,
    dag_builder=dag_builder,
    prerequisite_extractor=prerequisite_extractor
)


@router.post("/generate", response_model=LearningPathResponse)
async def generate_learning_path(payload: LearningPathRequest) -> LearningPathResponse:
    """
    Generate a personalized learning roadmap using DAG-based sequencing.
    
    Takes skills to learn and their metadata.
    Returns an optimized learning sequence with DAG structure and parallel tracks.
    
    Prerequisites can be:
    1. Provided in skill_metadata (priority)
    2. Auto-extracted using LLM (if auto_extract_prerequisites=True)
    3. Inferred as empty (default)
    
    Example request:
    ```json
    {
        "skills_to_learn": ["Docker", "Kubernetes", "AWS"],
        "skill_metadata": {
            "Docker": {
                "difficulty": 6,
                "time": "2 weeks",
                "unlock_power": 8,
                "prerequisites": []
            },
            "Kubernetes": {
                "difficulty": 9,
                "time": "2-3 months",
                "unlock_power": 4,
                "prerequisites": ["Docker"]
            }
        },
        "auto_extract_prerequisites": true,
        "available_time_weeks": 12
    }
    ```
    """
    try:
        # Validate inputs
        if not payload.skill_metadata:
            raise ValueError("skill_metadata cannot be empty")
        
        if not payload.skills_to_learn:
            raise ValueError("skills_to_learn cannot be empty")
        
        # Generate learning path with DAG
        path = await path_generator.generate_path(
            skills_to_learn=payload.skills_to_learn,
            skill_metadata=payload.skill_metadata,
            auto_extract_prerequisites=payload.auto_extract_prerequisites,
            available_time_weeks=payload.available_time_weeks,
            max_difficulty_per_phase=payload.max_difficulty_per_phase or 8,
        )
        
        return path
        
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating learning path: {str(exc)}"
        ) from exc


@router.post("/score-skills")
async def score_skills(skill_metadata: dict):
    """
    Get priority scores for a set of skills without generating a full path.
    
    Useful for debugging and understanding prioritization.
    
    Example request:
    ```json
    {
        "Python": {"difficulty": 2, "time": "3 days", "unlock_power": 10},
        "Kubernetes": {"difficulty": 9, "time": "2-3 months", "unlock_power": 4}
    }
    ```
    """
    try:
        scored = scorer.score_all_skills(skill_metadata)
        return {
            'scores': [
                {
                    'name': s.name,
                    'score': s.score,
                    'difficulty': s.difficulty,
                    'time': s.time,
                    'unlock_power': s.unlock_power,
                }
                for s in scored
            ]
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        ) from exc


@router.post("/dag-visualize")
async def visualize_dag(payload: LearningPathRequest) -> dict:
    """
    Generate a text visualization of the DAG for debugging.
    
    Returns the DAG structure and a visual representation showing
    learning layers, dependencies, and priority scores.
    """
    try:
        path = await path_generator.generate_path(
            skills_to_learn=payload.skills_to_learn,
            skill_metadata=payload.skill_metadata,
            auto_extract_prerequisites=payload.auto_extract_prerequisites,
        )
        
        visualization = dag_builder.visualize_dag_text(path.dag)
        
        return {
            'dag': path.dag,
            'visualization': visualization,
            'learning_layers': dag_builder.get_learning_layers(path.dag)
        }
        
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        ) from exc


@router.get("/config")
async def get_config():
    """Get current learning path configuration."""
    return {
        'difficulty_exponent': learning_path_settings.difficulty_exponent,
        'time_weight': learning_path_settings.time_weight,
        'unlock_power_weight': learning_path_settings.unlock_power_weight,
        'max_difficulty_per_phase': learning_path_settings.max_difficulty_per_phase,
        'future_features': {
            'enable_learning_curve_adjustment': learning_path_settings.enable_learning_curve_adjustment,
            'enable_job_market_weighting': learning_path_settings.enable_job_market_weighting,
            'enable_spaced_repetition': learning_path_settings.enable_spaced_repetition,
            'enable_motivation_balancing': learning_path_settings.enable_motivation_balancing,
            'enable_parallel_path_optimization': learning_path_settings.enable_parallel_path_optimization,
        }
    }


@router.post("/graph")
async def get_graph(payload: LearningPathRequest) -> dict:
    """
    Generate lightweight DAG graph structure (nodes + edges).
    
    Perfect for frontend visualization libraries like D3.js, Cytoscape, Sigma.js, etc.
    
    Returns:
    ```json
    {
        "nodes": [
            {"id": "Python", "difficulty": 3, "score": 8.5, "normalized_score": 1.0},
            {"id": "ML", "difficulty": 7, "score": 4.2, "normalized_score": 0.45}
        ],
        "edges": [
            {"from": "Python", "to": "ML", "weight": 0.45, "type": "hard"}
        ]
    }
    ```
    
    Node fields:
    - id: Skill name
    - difficulty: Difficulty level (1-10)
    - score: Calculated priority score
    - normalized_score: Score normalized to 0-1 range
    - time: Estimated learning time
    - unlock_power: Impact/value of the skill
    - rank: Priority rank within same dependency level
    
    Edge fields:
    - from: Prerequisite skill
    - to: Dependent skill
    - weight: Normalized priority score of target skill (0-1)
    - type: Always "hard" for prerequisites
    """
    try:
        path = await path_generator.generate_path(
            skills_to_learn=payload.skills_to_learn,
            skill_metadata=payload.skill_metadata,
            auto_extract_prerequisites=payload.auto_extract_prerequisites,
            available_time_weeks=payload.available_time_weeks,
            max_difficulty_per_phase=payload.max_difficulty_per_phase or 8,
        )
        
        return {
            'graph': path.graph,
            'learning_sequence': path.learning_sequence,
            'tracks': path.tracks
        }
        
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        ) from exc


@router.post("/from-skill-gaps")
async def generate_from_skill_gaps(payload: dict) -> dict:
    """
    Generate learning path directly from /extract-skills API output.
    
    This endpoint integrates with the skill extraction API and handles:
    - Converting time format (days → weeks/days string)
    - Building learning path from skill gaps
    - Optionally filtering by confidence scores
    
    Request format (from /extract-skills):
    ```json
    {
      "skill_gaps": {
        "Python Programming": {
          "difficulty": 4,
          "time": 68,              // Days (integer)
          "unlock_power": 11,
          "prerequisites": []      // Optional
        },
        "Machine Learning": {
          "difficulty": 7,
          "time": 28,
          "unlock_power": 9
        }
      },
      "auto_extract_prerequisites": true,
      "available_time_weeks": 24,
      "max_difficulty_per_phase": 8
    }
    ```
    
    Returns:
    ```json
    {
      "learning_sequence": ["Python Programming", "Machine Learning", ...],
      "graph": {nodes: [...], edges: [...]},
      "tracks": {primary: [...], secondary: [...], warmup: [...]},
      "skill_stats": {
        "total_skills": int,
        "total_weeks": float,
        "avg_difficulty": float,
        "hardest_skill": str
      }
    }
    ```
    """
    try:
        # Extract skill gaps from payload
        skill_gaps = payload.get("skill_gaps", {})
        auto_extract_prerequisites = payload.get("auto_extract_prerequisites", True)
        available_time_weeks = payload.get("available_time_weeks")
        max_difficulty_per_phase = payload.get("max_difficulty_per_phase", 8)
        
        if not skill_gaps:
            raise ValueError("skill_gaps cannot be empty")
        
        # Convert skill gaps format to learning path format
        converted_metadata = SkillGapConverter.convert_extract_skills_to_learning_path(skill_gaps)
        skills_to_learn = list(skill_gaps.keys())
        
        # Get statistics
        skill_stats = SkillGapConverter.get_skill_stats(skill_gaps)
        
        # Generate learning path with converted format
        path = await path_generator.generate_path(
            skills_to_learn=skills_to_learn,
            skill_metadata=converted_metadata,
            auto_extract_prerequisites=auto_extract_prerequisites,
            available_time_weeks=available_time_weeks,
            max_difficulty_per_phase=max_difficulty_per_phase,
        )
        
        return {
            'learning_sequence': path.learning_sequence,
            'graph': path.graph,
            'tracks': path.tracks,
            'scored_skills': [
                {
                    'name': s.name,
                    'score': round(s.score, 2),
                    'difficulty': s.difficulty,
                    'time': s.time,
                    'unlock_power': s.unlock_power
                }
                for s in path.scored_skills
            ],
            'prerequisites_map': path.prerequisites_map,
            'skill_stats': skill_stats,
            'metadata': path.metadata
        }
        
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating learning path from skill gaps: {str(exc)}"
        ) from exc


@router.post("/convert-format")
async def convert_skill_format(payload: dict) -> dict:
    """
    Convert skill gaps format (from /extract-skills) to learning path format.
    
    Useful for testing format conversion and understanding the transformation.
    
    Request:
    ```json
    {
      "skill_gaps": {
        "Python Programming": {
          "difficulty": 4,
          "time": 68,
          "unlock_power": 11
        },
        "Machine Learning": {
          "difficulty": 7,
          "time": 28,
          "unlock_power": 9
        }
      }
    }
    ```
    
    Response:
    ```json
    {
      "original": {...},
      "converted": {
        "Python Programming": {
          "difficulty": 4,
          "time": "9.7 weeks",
          "unlock_power": 11
        },
        "Machine Learning": {
          "difficulty": 7,
          "time": "4 weeks",
          "unlock_power": 9
        }
      },
      "stats": {
        "total_skills": 2,
        "total_weeks": 13.7,
        "avg_difficulty": 5.5
      }
    }
    ```
    """
    try:
        skill_gaps = payload.get("skill_gaps", {})
        
        if not skill_gaps:
            raise ValueError("skill_gaps cannot be empty")
        
        # Convert format
        converted = SkillGapConverter.convert_extract_skills_to_learning_path(skill_gaps)
        stats = SkillGapConverter.get_skill_stats(skill_gaps)
        
        return {
            'original': skill_gaps,
            'converted': converted,
            'stats': stats,
            'notes': {
                'time_format': 'Converted from days (integer) to human-readable string',
                'example_conversions': {
                    '7': '1 week',
                    '14': '2 weeks',
                    '68': '9.7 weeks',
                    '90': '13 weeks',
                    '3': '3 days'
                }
            }
        }
        
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error converting format: {str(exc)}"
        ) from exc
