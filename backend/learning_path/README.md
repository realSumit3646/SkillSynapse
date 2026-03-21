# Learning Path Generation Module (DAG-Based)

## Overview

The learning path module generates personalized learning roadmaps based on:
- **Skills to learn** (provided by teammate)
- **Skill metadata** (difficulty, time, value)
- **Prerequisites** (provided or auto-extracted using LLM)

It calculates priority scores and creates a Directed Acyclic Graph (DAG), then generates an optimized learning sequence with parallel learning tracks.

---

## Architecture: From Input to DAG to Learning Path

```
INPUT
│
├─ Skills to Learn: ["Docker", "Kubernetes", "AWS"]
├─ Metadata: {difficulty, time, unlock_power, prerequisites?}
└─ Optional: auto_extract_prerequisites = True

       │
       ▼
PROCESSING PIPELINE
├─ [1] Score each skill by priority
│      Score = (unlock_power) / (difficulty^1.5 × time_weeks)
│
├─ [2] Extract prerequisites
│      • Use provided prerequisites OR
│      • Use LLM to auto-extract
│
├─ [3] Build Directed Acyclic Graph (DAG)
│      {skill: {prerequisites, dependents, score, rank}}
│
├─ [4] Topological sort by priority
│      Respects prerequisites while maintaining priority order
│
└─ [5] Create parallel learning tracks
       Primary, Secondary, Warmup

       │
       ▼
OUTPUT
└─ LearningPathResponse
   ├─ scored_skills: All skills with scores
   ├─ prerequisites_map: {skill: [prereqs]}
   ├─ dag: Full graph structure
   ├─ learning_sequence: Ordered list
   └─ tracks: Parallel execution paths
```

---

## Module Structure

```
backend/learning_path/
├── __init__.py                    # Module marker
├── models.py                      # Pydantic schemas
├── scorer.py                      # Priority scoring algorithm
├── prerequisite_extractor.py      # LLM-based prerequisite extraction
├── dag_builder.py                 # DAG construction & analysis
├── path_generator.py              # Complete path generation orchestration
├── config.py                      # Configuration & feature flags
├── README.md                      # This file
├── EXAMPLES.py                    # Usage examples
└── curl_examples.sh               # API testing

Routes exposed via:
└── backend/routes/learning_path.py
```

---

## Key Components Explained

### 1. **Scorer Service** (`scorer.py`)

**Purpose:** Calculate priority score for each skill

**Formula:**
```
Score = (unlock_power × unlock_power_weight) / 
        (difficulty^difficulty_exponent × time_weeks × time_weight)
```

**Interpretation:**
- **High score** → Learn first (valuable, quick, easy)
- **Low score** → Learn later (less valuable, long, hard)

**Example:**
```
Docker:      Score = 8 / (6^1.5 × 2) = 0.54 (Learn first)
Kubernetes:  Score = 4 / (9^1.5 × 10.7) = 0.014 (Learn last)
```

---

### 2. **Prerequisite Extractor** (`prerequisite_extractor.py`)

**Purpose:** Intelligently extract skill dependencies using LLM (BATCH PROCESSING)

**Optimization: Single API Call for All Skills**
Instead of calling LLM once per skill (rate limiting issues), we send ALL skills in a SINGLE request.

**How it works:**
1. Collect all skills needing prerequisite extraction
2. Send in ONE batch request to Gemini LLM
3. LLM analyzes all skills together
4. Returns complete dependency map for all skills
5. Validates prerequisites are in available skills list

**Three-Tier Strategy:**
```
Tier 1: Use provided prerequisites (fastest) ✓
   └─ If metadata includes prerequisites, use directly

Tier 2: BATCH Extract via LLM (efficient) ✓
   └─ Single API call for all remaining skills
   └─ No rate limiting issues

Tier 3: Default to empty (safe fallback) ✓
   └─ If LLM fails, safely default to no prerequisites
```

**Example - Single Request for All Skills:**
```
POST request to LLM:
{
  "skills_list": [
    "- Kubernetes",
    "- Terraform",
    "- AWS Advanced"
  ]
}

Single LLM Response (one request, all skills):
{
    "Kubernetes": ["Docker"],
    "Terraform": ["AWS"],
    "AWS Advanced": ["AWS"]
}
```

**Benefits:**
- ✅ Single API call (no rate limiting)
- ✅ Batch processing is 5-10x faster
- ✅ Lower API costs
- ✅ Graceful fallback on failure

---

### 3. **DAG Builder** (`dag_builder.py`)

**Purpose:** Build a Directed Acyclic Graph from prerequisites and scores

**DAG Structure:**
```python
{
    "Docker": {
        "prerequisites": [],              # What must be learned first
        "dependents": ["Kubernetes"],     # What depends on this
        "priority_score": 0.54,
        "priority_rank": 1,               # Rank among peers
        "difficulty": 6,
        "time": "2 weeks",
        "unlock_power": 8
    },
    "Kubernetes": {
        "prerequisites": ["Docker"],
        "dependents": [],
        "priority_score": 0.014,
        "priority_rank": 1,
        "difficulty": 9,
        "time": "2-3 months",
        "unlock_power": 4
    }
}
```

**Key Features:**
- ✅ Detects circular dependencies (raises error)
- ✅ Calculates priority ranks within dependency levels
- ✅ Generates learning layers (topological levels)
- ✅ Provides text visualization

**Learning Layers Example:**
```
Layer 0: ["Docker"]                 # No prerequisites
Layer 1: ["Kubernetes", "AWS"]      # Depends on Docker
Layer 2: ["Terraform"]              # Depends on AWS
```

---

### 4. **Path Generator** (`path_generator.py`)

**Purpose:** Orchestrate all services to create complete learning path

**Process:**
1. Score all skills
2. Extract/use prerequisites
3. Build DAG
4. Get topological sort by priority
5. Create parallel tracks
6. Apply time constraints (optional)

**Result:** Optimized learning sequence respecting dependencies

---

## API Endpoints

### 1. **Generate Learning Path with DAG**

```http
POST /learning-path/generate
Content-Type: application/json

{
  "skills_to_learn": ["Docker", "Kubernetes"],
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
  "auto_extract_prerequisites": false,
  "available_time_weeks": 12
}
```

**Response:**
```json
{
  "skills_to_learn": ["Docker", "Kubernetes"],
  "scored_skills": [...],
  "prerequisites_map": {
    "Docker": [],
    "Kubernetes": ["Docker"]
  },
  "dag": {...},
  "learning_sequence": ["Docker", "Kubernetes"],
  "tracks": {
    "primary": ["Docker", "Kubernetes"],
    "secondary": [],
    "warmup": []
  },
  "metadata": {...}
}
```

---

### 2. **Get Lightweight Graph for Visualization (NEW!)**

Perfect for frontend visualization with D3.js, Cytoscape, Sigma.js, Vis.js, etc.

```http
POST /learning-path/graph
Content-Type: application/json

{
  "skills_to_learn": ["Python", "Machine Learning", "Deep Learning"],
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
    }
  },
  "auto_extract_prerequisites": false
}
```

**Response:**
```json
{
  "graph": {
    "nodes": [
      {
        "id": "Python",
        "difficulty": 3,
        "score": 26.19,
        "normalized_score": 1.0,
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
      }
    ],
    "edges": [
      {
        "from": "Python",
        "to": "Machine Learning",
        "weight": 0.89,
        "type": "hard"
      },
      {
        "from": "Machine Learning",
        "to": "Deep Learning",
        "weight": 0.45,
        "type": "hard"
      }
    ]
  },
  "learning_sequence": ["Python", "Machine Learning", "Deep Learning"],
  "tracks": {
    "primary": ["Python", "Machine Learning", "Deep Learning"],
    "secondary": [],
    "warmup": []
  }
}
```

**Graph Format Explained:**
- **nodes**: Each skill with difficulty, scores (raw + normalized 0-1), time, unlock_power, rank
- **edges**: Dependencies with weight (normalized score 0-1) and type ("hard" for prerequisites)
- **normalized_score**: Use for coloring (green=high priority, red=low priority)
- **weight**: Use for edge thickness/opacity (thicker=higher priority target skill)

**Frontend Example (D3.js):**
```javascript
// Use normalized_score for color
const colorScale = d3.scaleLinear()
  .domain([0, 1])
  .range(['#d73027', '#1a9850'])  // Red to Green

nodes.attr('fill', d => colorScale(d.normalized_score))

// Use weight for edge width
links.attr('stroke-width', d => d.weight * 5)
```

---

### 3. **Visualize DAG**

```http
POST /learning-path/dag-visualize
Content-Type: application/json

{
  "skills_to_learn": ["Docker", "Kubernetes"],
  "skill_metadata": {...},
  "auto_extract_prerequisites": false
}
```

**Response:** Text visualization showing layers, dependencies, and priority order

---

### 4. **Score Skills**

```http
POST /learning-path/score-skills
Content-Type: application/json

{
  "Docker": {"difficulty": 6, "time": "2 weeks", "unlock_power": 8},
  "Kubernetes": {"difficulty": 9, "time": "2-3 months", "unlock_power": 4}
}
```

---

### 5. **Get Configuration**

```http
GET /learning-path/config
```

---

## How DAG Ensures Correct Learning Order

**Without DAG (Priority-only):**
```
Score-based sorting:
  Kubernetes: 8.0 (high)
  Docker: 2.0 (low)
  Result: ["Kubernetes", "Docker"] ❌ WRONG
  Problem: Kubernetes depends on Docker!
```

**With DAG (Priority + Dependencies):**
```
Topological sort with priority:
  Check prerequisites for each skill
  Docker: no prerequisites → can learn
  Kubernetes: requires Docker → wait
  Result: ["Docker", "Kubernetes"] ✅ CORRECT
  Respects dependencies while maintaining priority within levels
```

---

## Configuration

Set in `.env`:

```bash
# Scoring Algorithm
DIFFICULTY_EXPONENT=1.5
TIME_WEIGHT=1.0
UNLOCK_POWER_WEIGHT=1.0

# Learning Constraints
MAX_DIFFICULTY_PER_PHASE=8

# Future Features
ENABLE_LEARNING_CURVE_ADJUSTMENT=false
ENABLE_JOB_MARKET_WEIGHTING=false
ENABLE_SPACED_REPETITION=false
ENABLE_MOTIVATION_BALANCING=false
ENABLE_PARALLEL_PATH_OPTIMIZATION=false
```

---

## Understanding the DAG Visualization

Example output from `/learning-path/dag-visualize`:

```
================================================================================
SKILL LEARNING DAG
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

📚 LAYER 2:
────────────────────────────────────────────────────────────────────────────────
  ✓ Terraform
    Priority Score: 3.27 (Rank: 1)
    Difficulty: 5/10
    Time: 1 week
    Prerequisites: AWS
    Required for: None

================================================================================
LEARNING ORDER
================================================================================
1. Docker (Score: 5.41)
2. AWS (Score: 4.12)
3. Terraform (Score: 3.27)
4. Kubernetes (Score: 0.47)
================================================================================
```

---

## Prerequisite Extraction Strategy

### **Option 1: Provided Prerequisites**
```python
"Docker": {
  "difficulty": 6,
  "prerequisites": ["Linux"]  # ← Explicitly provided
}
```

### **Option 2: Auto-Extracted by LLM**
```python
# Request with auto_extract_prerequisites=True
"Docker": {
  "difficulty": 6
  # LLM will determine prerequisites
}

# System sends to LLM:
# "What are the prerequisites for Docker?
#  Available skills: [Python, Linux, Networking, ...]"
# 
# LLM response: ["Linux"]
```

### **Option 3: Default (No Prerequisites)**
```python
"Docker": {
  "difficulty": 6
  # No prerequisites provided or extracted
  # Defaults to empty list
}
```

---

## Future Enhancements

All built-in feature flags ready to implement:

1. **Learning Curve Adjustment**
   - Adjust estimated times based on user's learning speed
   - Quick learner? Reduce time estimates

2. **Job Market Weighting**
   - Pull current job postings
   - Increase priority of hot-demand skills

3. **Spaced Repetition**
   - Schedule review cycles for retention
   - Revisit foundational skills periodically

4. **Motivation Balancing**
   - Prevent overload (max 1 hard skill/month)
   - Ensure early wins

5. **Parallel Optimization**
   - Better track assignment algorithm
   - Minimize inter-track dependencies

---

## Usage Example

### API
```bash
curl -X POST http://localhost:8000/learning-path/generate \
  -H "Content-Type: application/json" \
  -d '{
    "skills_to_learn": ["Docker", "Kubernetes"],
    "skill_metadata": {
      "Docker": {"difficulty": 6, "time": "2 weeks", "unlock_power": 8},
      "Kubernetes": {"difficulty": 9, "time": "2-3 months", "unlock_power": 4, "prerequisites": ["Docker"]}
    }
  }'
```

### Python (Direct)
```python
from backend.learning_path.path_generator import LearningPathGenerator

generator = LearningPathGenerator()
path = await generator.generate_path(
    skills_to_learn=["Docker", "Kubernetes"],
    skill_metadata={...},
    auto_extract_prerequisites=True
)

print(path.learning_sequence)  # ["Docker", "Kubernetes"]
print(path.dag)                # Full graph structure
```

---

## FAQ

**Q: What if prerequisites form a cycle?**
A: DAG builder detects this and raises `ValueError: "Detected circular dependency..."`

**Q: Can I have skills with multiple prerequisites?**
A: Yes! Skill A can depend on [B, C, D]. System ensures all are learned first.

**Q: What happens if auto_extract_prerequisites fails?**
A: System catches exception, logs warning, uses empty prerequisite list.

**Q: Why priority rank within layers?**
A: Shows which skills are more important within the same dependency level. Helps with scheduling.

**Q: Can I override LLM-extracted prerequisites?**
A: Yes! Provided prerequisites in metadata take priority over LLM extraction.

**Q: How are parallel tracks created?**
A: Hard skills (diff ≥ threshold) → Primary  
Medium → Secondary  
Easy (diff ≤ 3) → Warmup  
Allows parallel learning with mixed difficulty.

---

## Testing

See [EXAMPLES.py](./EXAMPLES.py) and [curl_examples.sh](./curl_examples.sh) for comprehensive examples.

Run curl examples:
```bash
bash backend/learning_path/curl_examples.sh
```

