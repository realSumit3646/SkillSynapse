"""Learning Path Generation Service with DAG-based Sequencing."""

from typing import List, Dict
from backend.learning_path.models import SkillWithScore, LearningPathResponse
from backend.learning_path.scorer import ScoringService
from backend.learning_path.dag_builder import DAGBuilder
from backend.learning_path.prerequisite_extractor import PrerequisiteExtractor


class LearningPathGenerator:
    """Generates optimized learning paths using DAG-based sequencing."""
    
    def __init__(
        self,
        scorer: ScoringService = None,
        dag_builder: DAGBuilder = None,
        prerequisite_extractor: PrerequisiteExtractor = None
    ):
        """Initialize with optional custom services."""
        self.scorer = scorer or ScoringService()
        self.dag_builder = dag_builder or DAGBuilder()
        self.prerequisite_extractor = prerequisite_extractor or PrerequisiteExtractor()
    
    async def generate_path(
        self,
        skills_to_learn: List[str],
        skill_metadata: Dict[str, dict],
        auto_extract_prerequisites: bool = True,
        available_time_weeks: int = None,
        max_difficulty_per_phase: int = 8,
    ) -> LearningPathResponse:
        """
        Generate a complete learning roadmap using DAG.
        
        Args:
            skills_to_learn: Skills that need to be learned
            skill_metadata: Metadata for scoring each skill
            auto_extract_prerequisites: Use LLM to extract prerequisites if not provided
            available_time_weeks: Total time budget (optional)
            max_difficulty_per_phase: Max difficulty to learn in parallel
            
        Returns:
            LearningPathResponse with DAG and learning sequence
        """
        # Step 1: Score all skills
        scored_skills = self.scorer.score_all_skills(skill_metadata)
        
        # Step 2: Extract prerequisites (from metadata or LLM)
        prerequisites_map = await self._get_prerequisites(
            skills_to_learn,
            skill_metadata,
            auto_extract_prerequisites
        )
        
        # Step 3: Build DAG from scored skills and prerequisites
        dag = self.dag_builder.build_dag(scored_skills, prerequisites_map)
        
        # Step 4: Get topological sort ordered by priority
        learning_sequence = self.dag_builder.get_topological_sort_by_priority(dag)
        
        # Step 5: Create parallel learning tracks
        tracks = self._create_tracks(
            learning_sequence,
            dag,
            max_difficulty_per_phase
        )
        
        # Step 6: Apply time constraint (optional)
        metadata = {}
        if available_time_weeks:
            total_weeks = self._calculate_total_time(scored_skills)
            metadata['total_weeks_needed'] = total_weeks
            metadata['time_constraint_met'] = total_weeks <= available_time_weeks
            if total_weeks > available_time_weeks:
                learning_sequence = self._apply_time_constraint(
                    learning_sequence,
                    available_time_weeks,
                    skill_metadata
                )
                metadata['note'] = "Some skills deferred due to time constraint"
        
        # Step 7: Generate lightweight graph structure for visualization
        graph = self.dag_builder.get_graph_json(dag)
        
        return LearningPathResponse(
            skills_to_learn=skills_to_learn,
            scored_skills=scored_skills,
            prerequisites_map=prerequisites_map,
            dag=dag,
            graph=graph,
            learning_sequence=learning_sequence,
            tracks=tracks,
            metadata=metadata,
        )
    
    async def _get_prerequisites(
        self,
        skills_to_learn: List[str],
        skill_metadata: Dict[str, dict],
        auto_extract: bool
    ) -> Dict[str, List[str]]:
        """
        Get prerequisites for all skills using BATCH LLM processing.
        
        Three-tier strategy (OPTIMIZED for single API call):
        1. Use provided prerequisites from skill_metadata - FASTEST
        2. Extract ALL remaining skills via SINGLE LLM CALL - EFFICIENT (BATCH)
        3. Default to empty list - SAFE FALLBACK
        
        KEY OPTIMIZATION: 
        - Single API call for ALL skills instead of one per skill
        - Avoids rate limiting issues
        - Much more efficient than sequential extraction
        """
        prerequisites_map = {}
        skills_needing_extraction = []
        
        # Tier 1: Collect all provided prerequisites
        for skill in skills_to_learn:
            metadata = skill_metadata.get(skill, {})
            provided_prereqs = metadata.get('prerequisites', [])
            
            if provided_prereqs:
                prerequisites_map[skill] = provided_prereqs
            else:
                skills_needing_extraction.append(skill)
        
        # Tier 2: Extract ALL remaining skills in SINGLE BATCH CALL (if enabled)
        if auto_extract and self.prerequisite_extractor.chain and skills_needing_extraction:
            try:
                # SINGLE API CALL for all skills that need extraction
                batch_extracted = await self.prerequisite_extractor.extract_all_prerequisites_batch(
                    skills_needing_extraction
                )
                prerequisites_map.update(batch_extracted)
            except Exception as e:
                print(f"Warning: LLM batch extraction failed: {e}")
                # Fallback to empty for any missing skills
                for skill in skills_needing_extraction:
                    if skill not in prerequisites_map:
                        prerequisites_map[skill] = []
        else:
            # Tier 3: Default to empty for remaining skills
            for skill in skills_needing_extraction:
                prerequisites_map[skill] = []
        
        return prerequisites_map
    
    @staticmethod
    def _create_tracks(
        learning_sequence: List[str],
        dag: Dict[str, dict],
        max_difficulty_per_phase: int,
    ) -> Dict[str, List[str]]:
        """
        Create parallel learning tracks based on DAG layers.
        
        Strategy:
        - Primary: Top priority skills (highest scores)
        - Secondary: Medium priority skills
        - Warmup: Easy skills for momentum
        """
        tracks = {
            'primary': [],
            'secondary': [],
            'warmup': [],
        }
        
        for idx, skill_name in enumerate(learning_sequence):
            node = dag.get(skill_name)
            if not node:
                continue
            
            difficulty = node['difficulty']
            score = node['priority_score']
            
            # Assign to tracks
            if idx == 0:
                # First (highest priority) goes to primary
                tracks['primary'].append(skill_name)
            elif difficulty >= max_difficulty_per_phase:
                # Hard skills go to primary (sequential)
                tracks['primary'].append(skill_name)
            elif difficulty <= 3:
                # Easy skills for warm-up
                tracks['warmup'].append(skill_name)
            else:
                # Medium difficulty → secondary
                tracks['secondary'].append(skill_name)
        
        return tracks
    
    @staticmethod
    def _calculate_total_time(scored_skills: List[SkillWithScore]) -> float:
        """Calculate total weeks needed to learn all skills."""
        scorer = ScoringService()
        total_days = sum(
            scorer.parse_time_to_days(skill.time)
            for skill in scored_skills
        )
        return round(total_days / 7, 1)
    
    @staticmethod
    def _apply_time_constraint(
        learning_sequence: List[str],
        available_time_weeks: int,
        skill_metadata: Dict[str, dict],
    ) -> List[str]:
        """
        Filter learning sequence to fit within time constraint.
        
        Keeps highest-priority skills that fit in available time.
        """
        scorer = ScoringService()
        constrained_sequence = []
        total_time = 0
        
        for skill_name in learning_sequence:
            skill_data = skill_metadata.get(skill_name, {})
            time_days = scorer.parse_time_to_days(skill_data.get('time', '1 week'))
            time_weeks = time_days / 7
            
            if total_time + time_weeks <= available_time_weeks:
                constrained_sequence.append(skill_name)
                total_time += time_weeks
        
        return constrained_sequence
