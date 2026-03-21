"""
Example: How to use the Learning Path Generation System with DAG

This file demonstrates the API usage and internal service usage.
"""

# Example 1: Using the API endpoint with DAG
# =============================================
# This would be a cURL or client request to: POST /learning-path/generate

example_request_payload = {
    "skills_to_learn": ["Docker", "Kubernetes", "AWS", "Terraform"],
    "skill_metadata": {
        "Docker": {
            "difficulty": 6,
            "time": "2 weeks",
            "unlock_power": 8,
            "description": "Containerization"
        },
        "Kubernetes": {
            "difficulty": 9,
            "time": "2-3 months",
            "unlock_power": 4,
            "prerequisites": ["Docker"],  # Optional: provide or let LLM extract
            "description": "Container orchestration"
        },
        "AWS": {
            "difficulty": 7,
            "time": "3-4 weeks",
            "unlock_power": 7,
            "prerequisites": ["Docker"],
            "description": "Cloud platform"
        },
        "Terraform": {
            "difficulty": 5,
            "time": "1 week",
            "unlock_power": 6,
            "prerequisites": ["AWS"],
            "description": "Infrastructure as Code"
        }
    },
    "auto_extract_prerequisites": True,  # Let LLM extract if not provided
    "available_time_weeks": 16,
    "max_difficulty_per_phase": 8
}

# Expected Response (with DAG structure):
example_response = {
    "skills_to_learn": ["Docker", "Kubernetes", "AWS", "Terraform"],
    "scored_skills": [
        {
            "name": "Docker",
            "difficulty": 6,
            "time": "2 weeks",
            "unlock_power": 8,
            "score": 5.41,
            "prerequisites": []
        },
        {
            "name": "AWS",
            "difficulty": 7,
            "time": "3-4 weeks",
            "unlock_power": 7,
            "score": 4.12
        },
        {
            "name": "Terraform",
            "difficulty": 5,
            "time": "1 week",
            "unlock_power": 6,
            "score": 3.27
        },
        {
            "name": "Kubernetes",
            "difficulty": 9,
            "time": "2-3 months",
            "unlock_power": 4,
            "score": 0.47
        }
    ],
    "prerequisites_map": {
        "Docker": [],
        "Kubernetes": ["Docker"],
        "AWS": ["Docker"],
        "Terraform": ["AWS"]
    },
    "dag": {
        "Docker": {
            "prerequisites": [],
            "dependents": ["Kubernetes", "AWS"],
            "priority_score": 5.41,
            "priority_rank": 1,
            "difficulty": 6,
            "time": "2 weeks",
            "unlock_power": 8
        },
        "AWS": {
            "prerequisites": ["Docker"],
            "dependents": ["Terraform"],
            "priority_score": 4.12,
            "priority_rank": 1,
            "difficulty": 7,
            "time": "3-4 weeks",
            "unlock_power": 7
        },
        "Terraform": {
            "prerequisites": ["AWS"],
            "dependents": [],
            "priority_score": 3.27,
            "priority_rank": 1,
            "difficulty": 5,
            "time": "1 week",
            "unlock_power": 6
        },
        "Kubernetes": {
            "prerequisites": ["Docker"],
            "dependents": [],
            "priority_score": 0.47,
            "priority_rank": 2,
            "difficulty": 9,
            "time": "2-3 months",
            "unlock_power": 4
        }
    },
    "learning_sequence": ["Docker", "AWS", "Terraform", "Kubernetes"],
    "tracks": {
        "primary": ["Docker", "AWS", "Kubernetes"],
        "secondary": ["Terraform"],
        "warmup": []
    },
    "metadata": {
        "total_weeks_needed": 15.5,
        "time_constraint_met": True
    }
}


# Example 2: Using Services Directly in Python
# =============================================

from backend.learning_path.models import SkillMetadata, ScoringConfig
from backend.learning_path.scorer import ScoringService
from backend.learning_path.path_generator import LearningPathGenerator
from backend.learning_path.dag_builder import DAGBuilder

# Initialize services
custom_config = ScoringConfig(
    difficulty_exponent=1.5,
    time_weight=1.0,
    unlock_power_weight=1.0,
)
scorer = ScoringService(config=custom_config)
dag_builder = DAGBuilder()
path_generator = LearningPathGenerator(scorer=scorer, dag_builder=dag_builder)

# Test scoring a single skill
skill = SkillMetadata(
    name="Kubernetes",
    difficulty=9,
    time="2-3 months",
    unlock_power=4,
    prerequisites=["Docker"]
)

score = scorer.calculate_score(skill)
print(f"Score for Kubernetes: {score}")
print(scorer.get_score_explanation(skill))

# Score multiple skills
metadata = {
    "Docker": {"difficulty": 6, "time": "2 weeks", "unlock_power": 8},
    "Kubernetes": {"difficulty": 9, "time": "2-3 months", "unlock_power": 4},
    "AWS": {"difficulty": 7, "time": "3-4 weeks", "unlock_power": 7},
}

scored = scorer.score_all_skills(metadata)
for skill in scored:
    print(f"{skill.name}: {skill.score}")

# Build DAG
prerequisites_map = {
    "Docker": [],
    "Kubernetes": ["Docker"],
    "AWS": ["Docker"]
}

dag = dag_builder.build_dag(scored, prerequisites_map)
print("\n=== DAG Structure ===")
for skill_name, node in dag.items():
    print(f"{skill_name}:")
    print(f"  Prerequisites: {node['prerequisites']}")
    print(f"  Dependents: {node['dependents']}")
    print(f"  Priority Score: {node['priority_score']}")

# Get learning layers (topological levels)
layers = dag_builder.get_learning_layers(dag)
print("\n=== Learning Layers ===")
for layer_idx, layer in enumerate(layers):
    print(f"Layer {layer_idx}: {layer}")

# Get topological sort by priority
sequence = dag_builder.get_topological_sort_by_priority(dag)
print(f"\nRecommended Learning Order: {sequence}")

# Visualize DAG (text format)
visualization = dag_builder.visualize_dag_text(dag)
print(visualization)


# Example 3: Generate Complete Learning Path Async
# =================================================

import asyncio

async def generate_path_example():
    """Generate learning path asynchronously."""
    
    path = await path_generator.generate_path(
        skills_to_learn=["Docker", "Kubernetes", "AWS"],
        skill_metadata={
            "Docker": {"difficulty": 6, "time": "2 weeks", "unlock_power": 8},
            "Kubernetes": {"difficulty": 9, "time": "2-3 months", "unlock_power": 4, "prerequisites": ["Docker"]},
            "AWS": {"difficulty": 7, "time": "3-4 weeks", "unlock_power": 7, "prerequisites": ["Docker"]},
        },
        auto_extract_prerequisites=False,  # Use provided prerequisites
        available_time_weeks=16,
    )
    
    print(f"Skills to learn: {path.skills_to_learn}")
    print(f"Learning sequence: {path.learning_sequence}")
    print(f"Prerequisites map: {path.prerequisites_map}")
    print(f"Learning tracks: {path.tracks}")

# Run async function
asyncio.run(generate_path_example())


# Example 4: DAG Visualization Example
# ======================================

"""
Running visualize_dag_text() produces output like:

================================================================================
SKILL LEARNING DAG (Directed Acyclic Graph)
================================================================================

📚 LAYER 0 (Can learn in parallel):
────────────────────────────────────────────────────────────────────────────────

  ✓ Docker
    Priority Score: 5.41 (Rank: 1)
    Difficulty: 6/10
    Time: 2 weeks
    Prerequisites: None
    Required for: Kubernetes, AWS

📚 LAYER 1 (Can learn in parallel):
────────────────────────────────────────────────────────────────────────────────

  ✓ AWS
    Priority Score: 4.12 (Rank: 1)
    Difficulty: 7/10
    Time: 3-4 weeks
    Prerequisites: Docker
    Required for: Terraform

  ✓ Kubernetes
    Priority Score: 0.47 (Rank: 2)
    Difficulty: 9/10
    Time: 2-3 months
    Prerequisites: Docker
    Required for: None

📚 LAYER 2 (Can learn in parallel):
────────────────────────────────────────────────────────────────────────────────

  ✓ Terraform
    Priority Score: 3.27 (Rank: 1)
    Difficulty: 5/10
    Time: 1 week
    Prerequisites: AWS
    Required for: None

================================================================================
LEARNING ORDER (Respecting Prerequisites & Priority):
────────────────────────────────────────────────────────────────────────────────
1. Docker (Score: 5.41)
2. AWS (Score: 4.12)
3. Terraform (Score: 3.27)
4. Kubernetes (Score: 0.47)
================================================================================
"""


# Example 5: How DAG Respects Prerequisites
# ===========================================

"""
DAG ensures that even if Kubernetes has the HIGHEST score,
it won't be placed before Docker (its prerequisite).

Example:
- Kubernetes: score 8.0 (high)
- Docker: score 2.0 (low)
- Kubernetes prerequisites: ["Docker"]

Without DAG: ["Kubernetes", "Docker"]  ❌ WRONG
With DAG:   ["Docker", "Kubernetes"]   ✅ CORRECT

DAG respects dependencies while maintaining priority ordering
within each dependency level.
"""


# Example 6: Configuration & Tuning
# ===================================

"""
All scoring weights are configurable in .env:

DIFFICULTY_EXPONENT=1.5
TIME_WEIGHT=1.0
UNLOCK_POWER_WEIGHT=1.0

To change behavior:
1. Increase DIFFICULTY_EXPONENT → Penalize hard skills more
2. Increase TIME_WEIGHT → Penalize long skills more
3. Increase UNLOCK_POWER_WEIGHT → Reward valuable skills more

No code changes needed!
"""


# Example 7: Feature Preview - LLM Prerequisite Extraction
# ===========================================================

"""
When auto_extract_prerequisites=True:

1. System sends each skill to Gemini LLM
2. LLM responds with prerequisite list
3. System uses that to build DAG

Example:
- Skill: "Kubernetes"
- Available skills: ["Docker", "Python", "AWS", "Linux"]
- LLM response: ["Docker"]
- System uses this for DAG

This allows:
- No manual prerequisite mapping needed
- Intelligent skill relationship detection
- Flexibility for new skills
"""

# ============================================================================
# Example 8: BATCH LLM Extraction (OPTIMIZED - Single API Call)
# ============================================================================
"""
KEY OPTIMIZATION: Instead of calling LLM once per skill (rate limiting),
send ALL skills in SINGLE request to LLM.

This is much more efficient and avoids rate limiting issues.
"""

batch_extraction_example = {
    "scenario": "Team wants to learn cloud platform with multiple skills",
    "skills_to_learn": [
        "Docker",
        "Kubernetes", 
        "AWS",
        "Terraform",
        "CI/CD"
    ],
    "skill_metadata": {
        # NOTE: NO prerequisites provided - let LLM extract all at once
        "Docker": {
            "difficulty": 6,
            "time": "2 weeks",
            "unlock_power": 8,
        },
        "Kubernetes": {
            "difficulty": 9,
            "time": "2-3 months",
            "unlock_power": 4,
        },
        "AWS": {
            "difficulty": 7,
            "time": "3-4 weeks",
            "unlock_power": 7,
        },
        "Terraform": {
            "difficulty": 5,
            "time": "1 week",
            "unlock_power": 6,
        },
        "CI/CD": {
            "difficulty": 4,
            "time": "2 weeks",
            "unlock_power": 5,
        }
    },
    "auto_extract_prerequisites": True,  # BATCH extraction enabled
}

"""
BATCH LLM PROCESSING:

Instead of:
  ❌ Call 1: Extract prerequisites for Docker
  ❌ Call 2: Extract prerequisites for Kubernetes
  ❌ Call 3: Extract prerequisites for AWS
  ❌ Call 4: Extract prerequisites for Terraform
  ❌ Call 5: Extract prerequisites for CI/CD
  (5 API calls = potential rate limiting!)

We do:
  ✅ Single Call: Send all 5 skills to LLM at once
  
Example LLM Request (Single):
{
  "skills_list": [
    "- Docker",
    "- Kubernetes",
    "- AWS",
    "- Terraform",
    "- CI/CD"
  ]
}

Example LLM Response (All skills at once):
{
    "Docker": [],                    # No prerequisites
    "Kubernetes": ["Docker"],        # Depends on Docker
    "AWS": [],                       # Can learn independently
    "Terraform": ["AWS"],            # Depends on AWS
    "CI/CD": ["Docker", "Git"]       # Git not in available list, filtered out
}

BENEFITS:
✅ Single API call (5-10x faster)
✅ No rate limiting issues
✅ Lower API costs
✅ Batch processing efficiency
✅ All prerequisites extracted together
"""

batch_extraction_result = {
    "prerequisites_map": {
        "Docker": [],
        "Kubernetes": ["Docker"],
        "AWS": [],
        "Terraform": ["AWS"],
        "CI/CD": ["Docker"]
    },
    "dag": {
        "Docker": {
            "prerequisites": [],
            "dependents": ["Kubernetes", "CI/CD"],
            "priority_score": 0.54,
            "priority_rank": 1,
            "difficulty": 6,
            "time": "2 weeks",
            "unlock_power": 8
        },
        "AWS": {
            "prerequisites": [],
            "dependents": ["Terraform"],
            "priority_score": 0.45,
            "priority_rank": 2,
            "difficulty": 7,
            "time": "3-4 weeks",
            "unlock_power": 7
        },
        "Kubernetes": {
            "prerequisites": ["Docker"],
            "dependents": [],
            "priority_score": 0.014,
            "priority_rank": 1,
            "difficulty": 9,
            "time": "2-3 months",
            "unlock_power": 4
        },
        "Terraform": {
            "prerequisites": ["AWS"],
            "dependents": [],
            "priority_score": 0.37,
            "priority_rank": 1,
            "difficulty": 5,
            "time": "1 week",
            "unlock_power": 6
        },
        "CI/CD": {
            "prerequisites": ["Docker"],
            "dependents": [],
            "priority_score": 0.25,
            "priority_rank": 1,
            "difficulty": 4,
            "time": "2 weeks",
            "unlock_power": 5
        }
    },
    "learning_sequence": [
        "Docker",      # Highest priority, no deps
        "AWS",         # High priority, no deps
        "CI/CD",       # Depends on Docker (satisfied)
        "Terraform",   # Depends on AWS (satisfied)
        "Kubernetes"   # Depends on Docker (satisfied)
    ],
    "tracks": {
        "primary": ["Docker", "AWS"],
        "secondary": ["CI/CD", "Terraform"],
        "warmup": []
    },
    "metadata": {
        "total_weeks_needed": 16.3,
        "time_constraint_met": True,
        "optimization": "Batch LLM extraction - single API call for all 5 skills"
    }
}

# ============================================================================
# Example 9: Lightweight Graph Structure for Visualization (NEW!)
# ============================================================================
"""
The /learning-path/graph endpoint returns a lightweight nodes/edges format
perfect for D3.js, Cytoscape, Sigma.js, Vis.js and other graph visualization libraries.

This is ideal for:
- Frontend visualization
- Network analysis tools
- Graph-based UI components
- Mobile-friendly data transmission
"""

graph_request = {
    "skills_to_learn": ["Python", "Machine Learning", "Deep Learning", "Docker"],
    "skill_metadata": {
        "Python": {
            "difficulty": 3,
            "time": "3 weeks",
            "unlock_power": 10
        },
        "Machine Learning": {
            "difficulty": 7,
            "time": "8 weeks",
            "unlock_power": 9,
            "prerequisites": ["Python"]
        },
        "Deep Learning": {
            "difficulty": 9,
            "time": "10 weeks",
            "unlock_power": 8,
            "prerequisites": ["Machine Learning"]
        },
        "Docker": {
            "difficulty": 4,
            "time": "2 weeks",
            "unlock_power": 7
        }
    },
    "auto_extract_prerequisites": False
}

# Expected Response from POST /learning-path/graph:
graph_response = {
    "graph": {
        "nodes": [
            {
                "id": "Python",
                "difficulty": 3,
                "score": 26.19,
                "normalized_score": 1.0,        # Highest score (0-1 range)
                "time": "3 weeks",
                "unlock_power": 10,
                "rank": 1
            },
            {
                "id": "Machine Learning",
                "difficulty": 7,
                "score": 3.21,
                "normalized_score": 0.89,
                "time": "8 weeks",
                "unlock_power": 9,
                "rank": 1
            },
            {
                "id": "Deep Learning",
                "difficulty": 9,
                "score": 1.52,
                "normalized_score": 0.45,
                "time": "10 weeks",
                "unlock_power": 8,
                "rank": 1
            },
            {
                "id": "Docker",
                "difficulty": 4,
                "score": 10.75,
                "normalized_score": 0.92,
                "time": "2 weeks",
                "unlock_power": 7,
                "rank": 1
            }
        ],
        "edges": [
            {
                "from": "Python",
                "to": "Machine Learning",
                "weight": 0.89,                 # Normalized score of ML
                "type": "hard"
            },
            {
                "from": "Machine Learning",
                "to": "Deep Learning",
                "weight": 0.45,                 # Normalized score of DL
                "type": "hard"
            }
        ]
    },
    "learning_sequence": ["Python", "Docker", "Machine Learning", "Deep Learning"],
    "tracks": {
        "primary": ["Python", "Machine Learning", "Deep Learning"],
        "secondary": ["Docker"],
        "warmup": []
    }
}

"""
GRAPH STRUCTURE EXPLANATION:

Nodes:
------
- id: Skill name (unique identifier)
- difficulty: Difficulty level (1-10)
- score: Calculated priority score (raw value)
- normalized_score: Score normalized to 0-1 range (for visualization coloring/sizing)
- time: Estimated learning time
- unlock_power: Impact/value of the skill
- rank: Priority rank within same dependency level

Edges:
------
- from: Prerequisite skill (source)
- to: Dependent skill (target)
- weight: Normalized priority score of target skill (0-1, for visualization thickness/opacity)
- type: Always "hard" for prerequisites (extensible for soft dependencies)

USE IN FRONTEND:
- Use normalized_score to color nodes (high=green, low=red)
- Use weight to size edges (thicker=higher priority)
- Use edges to draw dependency arrows
- Use learning_sequence to animate learning order
- Use tracks to group skills into learning phases

EXAMPLE D3.js VISUALIZATION:
d3.force()
  .nodes(graph.nodes)
  .links(graph.edges)
  .on("tick", () => {
    // Color by normalized_score
    nodes.attr("fill", d => {
      const score = d.normalized_score
      return d3.interpolateRdYlGn(score)  // Red (low) to Green (high)
    })
    
    // Size by unlock_power
    nodes.attr("r", d => d.unlock_power / 2)
    
    // Width by weight
    links.attr("stroke-width", d => d.weight * 5)
  })
"""

