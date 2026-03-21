"""Directed Acyclic Graph (DAG) Builder for Learning Paths."""

from typing import Dict, List, Set, Tuple
from backend.learning_path.models import SkillWithScore


class DAGBuilder:
    """
    Builds a Directed Acyclic Graph from skills and their prerequisites.
    
    DAG Structure:
    {
        "Python": {
            "prerequisites": [],           # What must be learned first
            "dependents": ["ML", "Web"],   # What depends on this skill
            "priority_score": 8.5,
            "priority_rank": 1             # Rank among peers
        },
        "ML": {
            "prerequisites": ["Python"],
            "dependents": [],
            "priority_score": 4.2,
            "priority_rank": 2
        }
    }
    """
    
    def __init__(self):
        pass
    
    def build_dag(
        self,
        scored_skills: List[SkillWithScore],
        prerequisites_map: Dict[str, List[str]]
    ) -> Dict[str, dict]:
        """
        Build a DAG from scored skills and prerequisites.
        
        Args:
            scored_skills: List of SkillWithScore objects
            prerequisites_map: {skill: [prerequisite_skills]}
            
        Returns:
            DAG structure with graph information
        """
        # Initialize DAG nodes
        dag = {}
        skill_by_name = {skill.name: skill for skill in scored_skills}
        
        # Build reverse lookup (what depends on each skill)
        dependents_map: Dict[str, Set[str]] = {skill.name: set() for skill in scored_skills}
        
        for skill in scored_skills:
            prereqs = prerequisites_map.get(skill.name, [])
            
            # Add node to DAG
            dag[skill.name] = {
                "prerequisites": prereqs,
                "dependents": [],  # Will be populated later
                "priority_score": skill.score,
                "difficulty": skill.difficulty,
                "time": skill.time,
                "unlock_power": skill.unlock_power,
                "priority_rank": 0,  # Will be calculated
            }
            
            # Build dependents map
            for prereq in prereqs:
                if prereq in skill_by_name:
                    dependents_map[prereq].add(skill.name)
        
        # Add dependents to DAG
        for skill_name, dependents in dependents_map.items():
            if skill_name in dag:
                dag[skill_name]["dependents"] = sorted(list(dependents))
        
        # Validate DAG (check for cycles)
        if self._has_cycle(dag):
            raise ValueError("Detected circular dependency in prerequisites. Cannot create valid DAG.")
        
        # Calculate priority ranks (within same dependency level)
        self._calculate_priority_ranks(dag)
        
        return dag
    
    def _has_cycle(self, dag: Dict[str, dict]) -> bool:
        """
        Detect if DAG has cycles using DFS.
        
        Returns True if cycle detected, False otherwise.
        """
        visited = set()
        rec_stack = set()  # Recursion stack
        
        def has_cycle_dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            # Check all nodes that depend on this one
            for dependent in dag.get(node, {}).get("dependents", []):
                if dependent not in visited:
                    if has_cycle_dfs(dependent):
                        return True
                elif dependent in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Run DFS from each unvisited node
        for node in dag:
            if node not in visited:
                if has_cycle_dfs(node):
                    return True
        
        return False
    
    def _calculate_priority_ranks(self, dag: Dict[str, dict]) -> None:
        """
        Calculate priority rank within dependency levels.
        
        Higher rank = higher priority within the same learning phase.
        Rank is relative to skills with the same prerequisites.
        """
        # Group skills by their prerequisites
        by_prerequisites: Dict[frozenset, List[str]] = {}
        
        for skill_name, node in dag.items():
            prereq_key = frozenset(node["prerequisites"])
            if prereq_key not in by_prerequisites:
                by_prerequisites[prereq_key] = []
            by_prerequisites[prereq_key].append(skill_name)
        
        # Sort within each group by priority score
        for prereq_group, skills in by_prerequisites.items():
            sorted_skills = sorted(
                skills,
                key=lambda s: dag[s]["priority_score"],
                reverse=True
            )
            for rank, skill_name in enumerate(sorted_skills, 1):
                dag[skill_name]["priority_rank"] = rank
    
    def get_learning_layers(self, dag: Dict[str, dict]) -> List[List[str]]:
        """
        Get learning layers (topological levels) from DAG.
        
        Returns list of learning phases where each phase contains
        skills that can be learned in parallel.
        
        Example:
        [
            ["Python"],           # Layer 0: No prerequisites
            ["Docker", "ML"],     # Layer 1: Only Python prerequisite
            ["Kubernetes"]        # Layer 2: Docker prerequisite
        ]
        """
        layers = []
        learned = set()
        remaining = set(dag.keys())
        
        max_iterations = len(dag) + 1
        iteration = 0
        
        while remaining and iteration < max_iterations:
            iteration += 1
            current_layer = []
            
            # Find all skills whose prerequisites are all learned
            for skill in sorted(remaining):
                prerequisites = dag[skill]["prerequisites"]
                if all(p in learned for p in prerequisites):
                    current_layer.append(skill)
            
            if not current_layer:
                # Shouldn't happen if DAG is valid, but safety check
                current_layer = sorted(list(remaining))
            
            # Sort by priority score within layer
            current_layer.sort(
                key=lambda s: dag[s]["priority_score"],
                reverse=True
            )
            
            layers.append(current_layer)
            learned.update(current_layer)
            remaining -= set(current_layer)
        
        return layers
    
    def get_topological_sort_by_priority(
        self,
        dag: Dict[str, dict]
    ) -> List[str]:
        """
        Get topological sort of DAG, ordered by priority score within levels.
        
        Returns a list of skills ordered for learning, respecting
        prerequisite dependencies and prioritizing by score.
        """
        layers = self.get_learning_layers(dag)
        result = []
        for layer in layers:
            result.extend(layer)
        return result
    
    def visualize_dag_text(self, dag: Dict[str, dict]) -> str:
        """
        Generate text visualization of DAG.
        
        Useful for debugging and understanding dependencies.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("SKILL LEARNING DAG (Directed Acyclic Graph)")
        lines.append("=" * 80)
        
        # Get layers for visualization
        layers = self.get_learning_layers(dag)
        
        for layer_idx, layer in enumerate(layers):
            lines.append(f"\n📚 LAYER {layer_idx} (Can learn in parallel):")
            lines.append("-" * 80)
            
            for skill in layer:
                node = dag[skill]
                prereqs_text = ", ".join(node["prerequisites"]) if node["prerequisites"] else "None"
                dependents_text = ", ".join(node["dependents"]) if node["dependents"] else "None"
                
                lines.append(f"\n  ✓ {skill}")
                lines.append(f"    Priority Score: {node['priority_score']:.2f} (Rank: {node['priority_rank']})")
                lines.append(f"    Difficulty: {node['difficulty']}/10")
                lines.append(f"    Time: {node['time']}")
                lines.append(f"    Prerequisites: {prereqs_text}")
                lines.append(f"    Required for: {dependents_text}")
        
        lines.append("\n" + "=" * 80)
        lines.append("LEARNING ORDER (Respecting Prerequisites & Priority):")
        lines.append("-" * 80)
        
        sequence = self.get_topological_sort_by_priority(dag)
        for idx, skill in enumerate(sequence, 1):
            node = dag[skill]
            lines.append(f"{idx}. {skill} (Score: {node['priority_score']:.2f})")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def get_graph_json(self, dag: Dict[str, dict]) -> Dict[str, list]:
        """
        Convert DAG to lightweight JSON graph format (nodes + edges).
        
        Perfect for frontend visualization, D3.js, Cytoscape, etc.
        
        Returns:
        {
            "nodes": [
                {"id": "Python", "difficulty": 3, "score": 8.5},
                {"id": "ML", "difficulty": 7, "score": 4.2}
            ],
            "edges": [
                {"from": "Python", "to": "ML", "weight": 0.85, "type": "hard"}
            ]
        }
        
        Args:
            dag: The built DAG from build_dag()
            
        Returns:
            {nodes, edges} structure for lightweight transmission
        """
        nodes = []
        edges = []
        
        # Normalize scores for weights (0.0 to 1.0)
        if dag:
            scores = [node["priority_score"] for node in dag.values()]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max_score - min_score if max_score != min_score else 1.0
        else:
            max_score = min_score = score_range = 1.0
        
        # Build nodes
        for skill_name, node in dag.items():
            normalized_score = (node["priority_score"] - min_score) / score_range if score_range > 0 else 0.5
            
            nodes.append({
                "id": skill_name,
                "difficulty": node["difficulty"],
                "score": round(node["priority_score"], 2),
                "normalized_score": round(normalized_score, 2),
                "time": node["time"],
                "unlock_power": node["unlock_power"],
                "rank": node["priority_rank"]
            })
        
        # Build edges (from prerequisites to dependent skills)
        added_edges = set()
        for skill_name, node in dag.items():
            for prerequisite in node["prerequisites"]:
                edge_id = (prerequisite, skill_name)
                if edge_id not in added_edges:
                    # Weight is the normalized priority score of the target skill
                    target_score = dag[skill_name]["priority_score"]
                    weight = (target_score - min_score) / score_range if score_range > 0 else 0.5
                    
                    edges.append({
                        "from": prerequisite,
                        "to": skill_name,
                        "weight": round(weight, 2),
                        "type": "hard"
                    })
                    added_edges.add(edge_id)
        
        return {
            "nodes": nodes,
            "edges": edges
        }
