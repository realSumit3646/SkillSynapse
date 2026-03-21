SkillSynapse - Adaptive Skill Learning Path Generator

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture & Design](#architecture--design)
4. [Technology Stack](#technology-stack)
5. [API Routes & Endpoints](#api-routes--endpoints)
6. [Service Layer Components](#service-layer-components)
7. [Algorithms & Formulas](#algorithms--formulas)
8. [Data Models & Schemas](#data-models--schemas)
9. [Configuration & Environment Setup](#configuration--environment-setup)
10. [Installation & Setup Instructions](#installation--setup-instructions)
11. [Usage Examples](#usage-examples)
12. [Deployment Guide](#deployment-guide)
13. [Troubleshooting & Support](#troubleshooting--support)

---

## Executive Summary

**SkillSynapse** is an adaptive skill learning path generation system that leverages AI/ML to transform job descriptions and resumes into personalized, prioritized learning roadmaps. The platform intelligently identifies skill gaps, estimates proficiency levels based on evidence from resume text, and constructs optimal prerequisite-aware learning sequences.

**Key Features:**
- **Skill Extraction**: Constraint-based LLM extraction of explicit skills from job descriptions
- **Proficiency Analysis**: Multi-indicator evidence-based scoring with semantic validation
- **Gap Identification**: Comparative analysis between required and available skills
- **Learning Path Generation**: DAG-based prerequisite-aware prioritization with parallel track identification
- **Resource Aggregation**: Automated collection of learning materials from Wikipedia, arXiv, academic sources, and web
- **Feedback Loop**: Stateful recomputation with user proficiency feedback refinement

**Target Users:**
- Job seekers transitioning to new roles
- Career development professionals
- Upskilling coordinators
- Individual learners seeking personalized roadmaps

---

## System Overview

### Problem Context

Professionals face challenges when:
1. Identifying exactly which skills are required for a target role
2. Assessing how their existing skills align with job requirements
3. Understanding which skills should be learned first (prerequisites)
4. Estimating time and difficulty for skill acquisition
5. Finding quality learning resources for specific skill transitions

SkillSynapse automates this entire workflow through a unified API.

### Solution Architecture

The system operates in four main flows:

**Flow 1: Skill Analysis (Analyze-Skills)**
- User uploads job description and resume
- System extracts required skills from JD
- System scores proficiency on existing skills from resume
- System identifies skill gaps and provides evidence
- Returns comprehensive skill proficiency matrix

**Flow 2: Feedback Loop (Provide-Feedback)**
- User reviews proficiency scores and provides corrections
- System stores state and recalculates scores with feedback weight
- Updated proficiency matrix returned without re-upload

**Flow 3: Learning Path Generation (Learning-Path)**
- Input: Skill gaps and requirements
- System builds prerequisite dependency graph (DAG)
- Calculates priority scores based on unlock unlocks, difficulty, and time
- Returns topologically ordered skills with parallel tracks

**Flow 4: Resource Discovery (Get-Resources)**
- Input: Skill transition pairs (from→to)
- Aggregates resources from 7+ public sources
- Categorizes by type (tutorial, documentation, course, forum)
- Returns URLs with fallback generation

---

## Architecture & Design

### Layered Architecture

```
┌─────────────────────────────────────┐
│        FastAPI Routes Layer         │
│  /analyze-skills, /provide-feedback │
│  /learning-path, /get-resources     │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      Service Layer                   │
│ ├─ SkillExtractor                    │
│ ├─ AnalysisService                   │
│ ├─ EmbeddingCluster                  │
│ ├─ PathGenerator                     │
│ └─ ResourceAggregator                │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│    ML/Utils/Config Layer             │
│ ├─ Gemini LLM (fallback chain)       │
│ ├─ Sentence-Transformers Embeddings  │
│ ├─ scikit-learn (clustering, scoring)│
│ ├─ PyMuPDF / python-docx (parsing)   │
│ └─ Config (environment mgmt)         │
└─────────────────────────────────────┘
```

### State Management

**AnalysisState Class** (in-memory):
```python
class AnalysisState:
    job_description: Optional[str]
    resume_text: Optional[str]
    required_skills: Optional[list[str]]
    evidence: Optional[dict[str, dict[str, float]]]
    sim_map: Optional[dict[str, float]]
    closest_map: Optional[dict[str, str | None]]
```

Stores intermediate results from `/analyze-skills` to enable stateful `/provide-feedback` recomputation without requiring file re-upload.

### Data Flow

```
[User Input] → [Parse Files] → [Extract Skills] → [Cluster Skills]
     ↓                                                    ↓
[Resume, JD]                                    [Reduced Skill Set]
                                                        ↓
                                        [Score Proficiency + Evidence]
                                                        ↓
                                        [Calculate Confidence/Difficulty]
                                                        ↓
                                        [Store in AnalysisState]
                                                        ↓
                                        [Return to User]
                                                        ↓
                                        [User Feedback] → [Recalculate with Feedback Weight]
```

---

## Technology Stack

### Backend Framework
- **FastAPI 0.116.1** - High-performance async web framework with automatic OpenAPI documentation
- **Pydantic 2.11.7** - Data validation and serialization with strict type checking
- **uvicorn** - ASGI application server for production deployment
- **pydantic-settings** - Environment-based configuration management

### LLM & NLP
- **LangChain Core & LangChain Google GenAI** - LLM orchestration with fallback chain support
- **Google Gemini** - Primary LLM for skill extraction, evidence scoring, prerequisite extraction
  - Primary: `gemini-3.1-flash-lite-preview`
  - Fallbacks: `gemini-2.5-flash`, `gemini-flash-latest`, `gemini-pro-latest`
- **sentence-transformers** (all-MiniLM-L6-v2) - Fast local embedding generation for semantic similarity
- **Optional: Gemini Embeddings** (text-embedding-004) - Configurable via settings

### Machine Learning & NLP Libraries
- **scikit-learn** - Cosine similarity, AgglomerativeClustering for semantic skill grouping
- **numpy** - Numerical computations and array operations
- **flashtext** - Fast keyword extraction for evidence span detection in resume
- **spaCy** (optional) - Advanced NLP for tokenization and dependency parsing

### Document Parsing
- **PyMuPDF (fitz) v1.26.4** - PDF extraction, text recovery, layout analysis
- **python-docx v1.2.0** - DOCX parsing, paragraph extraction
- **python-multipart v0.0.20** - Multipart form data handling for file uploads

### External APIs (Resource Discovery)
- Wikipedia API - Definitions and overviews
- arXiv API - Academic papers and research
- StackExchange API - Technical Q&A
- Google Books API - Book previews and references
- GitHub Search API - Code examples and repositories
- YouTube & Web Search - Video tutorials and blog posts

### Development Tools
- **pytest** - Unit testing framework
- **Black** - Code formatting
- **LLM Rate-Limit Resilience** - Model fallback chain for API quota management

---

## API Routes & Endpoints

### Route Group 1: Skill Analysis (`/skills`)

#### POST `/analyze-skills`
**Purpose:** Complete skill analysis workflow (extraction → scoring → gap identification)

**Request:**
```json
{
  "job_description": "string (required)",
  "resume_file": "file (required, PDF or DOCX)",
  "max_output_skills": "integer (optional, default 20)"
}
```

**Response:**
```json
{
  "required_skills": ["Python", "Machine Learning", "SQL", ...],
  "all_skills": [
    {
      "skill_name": "Python",
      "proficiency_score": 8.2,
      "confidence": 0.85,
      "evidence": "Mentioned in 'Data Analysis Project', 'Backend Development'",
      "similarity": 0.92
    },
    ...
  ],
  "skill_gaps": [
    {
      "skill_name": "Kubernetes",
      "required": true,
      "proficiency": 0,
      "difficulty": 7.1,
      "estimated_time_weeks": 4.2
    },
    ...
  ],
  "gap_count": 5,
  "coverage_percentage": 71.4
}
```

**Flow:**
1. Parse resume (PDF/DOCX) into text
2. Extract skills from job description using Gemini
3. Cluster extracted skills to reduce fragmentation
4. Score each skill against resume evidence
5. Calculate confidence and difficulty estimations
6. Identify gaps where proficiency < threshold
7. Store state in AnalysisState for feedback loop
8. Return comprehensive response

---

#### POST `/provide-feedback`
**Purpose:** Refine proficiency scores using user feedback (stateful)

**Request:**
```json
{
  "user_feedback": "string (JSON with skill-feedback pairs)"
}
```

**Example user_feedback:**
```json
{
  "Python": 9.5,
  "Machine Learning": 6.0,
  "Kubernetes": 3.5
}
```

**Response:** Same structure as `/analyze-skills` with updated scores

**Flow:**
1. Retrieve stored AnalysisState
2. Parse user feedback JSON
3. For each feedback item, apply blending: `score = 0.6 × score_model + 0.4 × user_feedback`
4. Recalculate confidence with user feedback weight
5. Return updated proficiency matrix

---

### Route Group 2: Learning Path Generation (`/learning-path`)

#### POST `/generate`
**Purpose:** Generate prerequisite-aware, priority-ordered learning sequence

**Request:**
```json
{
  "skills": [
    {
      "skill_name": "Kubernetes",
      "proficiency": 0,
      "difficulty": 7.1,
      "estimated_time_weeks": 4.2
    },
    ...
  ]
}
```

**Response:**
```json
{
  "learning_sequence": [
    {
      "skill_name": "Docker",
      "priority_score": 8.7,
      "difficulty": 6.5,
      "estimated_time_weeks": 2.5,
      "prerequisites": ["Linux Basics"],
      "level": 0,
      "track": "containerization"
    },
    ...
  ],
  "learning_layers": [
    ["Docker", "Basic Linux"],
    ["Kubernetes", "Docker Compose"],
    ["Advanced Orchestration"]
  ],
  "parallel_tracks": [
    "containerization",
    "orchestration",
    "cloud-deployment"
  ],
  "total_duration_weeks": 12.5,
  "graph": {
    "nodes": [...],
    "edges": [...]
  }
}
```

**Flow:**
1. Extract prerequisites for all skills (batch LLM call)
2. Build prerequisite DAG with cycle detection
3. Calculate priority scores using formula: `(unlock_power × w_u) / (difficulty^α × time_weeks × w_t)`
4. Perform topological sort with priority ranking within levels
5. Extract learning layers for parallel execution
6. Generate graph JSON for frontend visualization
7. Return complete learning sequence

---

#### POST `/from-skill-gaps`
**Purpose:** Convenience endpoint to feed `/analyze-skills` output directly to `/generate`

**Request:** SkillGapList from `/analyze-skills` response

**Response:** Same as `/generate`

---

#### GET `/graph/:skill_name`
**Purpose:** Fetch DAG visualization data for a specific skill's prerequisite tree

**Response:**
```json
{
  "nodes": [
    {"id": "Python", "label": "Python", "level": 0},
    {"id": "Data Structures", "label": "Data Structures", "level": 1},
    ...
  ],
  "edges": [
    {"from": "Python", "to": "Data Structures"},
    ...
  ]
}
```

---

### Route Group 3: Resource Discovery (`/skill-resources`)

#### POST `/get-resources`
**Purpose:** Aggregate learning materials for skill transitions

**Request:**
```json
{
  "transitions": [
    {
      "from_skill": "Python",
      "to_skill": "Data Science"
    },
    {
      "from_skill": "JavaScript",
      "to_skill": "React"
    }
  ]
}
```

**Response:**
```json
{
  "transitions": [
    {
      "from_skill": "Python",
      "to_skill": "Data Science",
      "resources": {
        "tutorials": [
          {"title": "...", "url": "...", "source": "YouTube"},
          ...
        ],
        "documentation": [
          {"title": "...", "url": "...", "source": "Official"},
          ...
        ],
        "courses": [
          {"title": "...", "url": "...", "source": "Coursera"},
          ...
        ],
        "research": [
          {"title": "...", "url": "...", "source": "arXiv"},
          ...
        ]
      }
    },
    ...
  ]
}
```

**Resource Types:**
- **tutorials** - Video, blog, interactive guides
- **documentation** - Official docs, API references
- **courses** - Structured learning paths
- **research** - Academic papers, whitepapers
- **forums** - Q&A communities (StackExchange)
- **books** - Books and textbooks
- **code** - GitHub repositories and examples

**Flow:**
1. For each skill transition pair
2. Query Wikipedia, arXiv, StackExchange, Google Books, GitHub, YouTube, web search
3. Categorize results by type
4. Rank by relevance (title match, source credibility)
5. Return top N per category
6. Fallback: Generate generic search links if APIs insufficient

---

## Service Layer Components

### 1. SkillExtractor (`backend/services/skill_extractor.py`)

**Purpose:** Constraint-based explicit skill extraction from job descriptions

**Key Method:** `extract_skills(job_description: str) → list[str]`

**Approach:**
- **Constraint:** Extracts only explicitly mentioned skills, not implicit requirements
- **Parsing:** Gemini LLM with strict prompt engineering
- **Output Format:** JSON array with fallback markdown code-block parsing
- **Post-processing:**
  - `is_explicit_in_text()` validation
  - Deduplication (case-insensitive)
  - display_name normalization (e.g., "c++" → "C++")
- **Fallback Models:** Chain of Gemini models with automatic retry

**Configuration:**
- Model: Configurable via `config.gemini_chat_model`
- Timeout: 30s per request
- Retry: Up to 3 models before failure

---

### 2. EmbeddingCluster (`backend/services/embedding_cluster.py`)

**Purpose:** Reduce skill fragmentation through semantic clustering

**Key Method:** `group_and_reduce(skills: list[str], max_output: int) → list[str]`

**Approach:**
- **Embedding:** sentence-transformers (all-MiniLM-L6-v2) or optional Gemini embeddings
- **Clustering:** Hierarchical agglomerative clustering (ward linkage)
- **Merging:** Iterative cluster merging until target count reached
- **Representative Selection:** LLM selects best representative per cluster (configurable)

**Configuration:**
- Max output skills: `config.max_output_skills` (default 20)
- Distance threshold: `config.cluster_distance_threshold` (default 0.38)
- Embedding model: `config.sentence_transformer_model` (default all-MiniLM-L6-v2)
- LLM naming: `config.allow_llm_cluster_naming` (toggle on/off)

**Algorithm:**
```
1. Embed all input skills
2. Build distance matrix (cosine similarity)
3. Agglomerative clustering until N clusters formed
4. While cluster_count > max_output:
     - Merge two closest clusters
     - Regenerate representative skill name
5. Return merged cluster representatives
```

---

### 3. AnalysisService (`backend/skill_proficiency/services/analysis_service.py`)

**Purpose:** Multi-indicator evidence-based proficiency scoring

**Key Methods:**
- `score_skill_proficiency(resume_text: str, skill: str) → SkillScore`
- `calculate_confidence(llm_confidence, indicators, mentions) → float`
- `estimate_skill_difficulty(similarity: float, related_skills: list) → float`
- `estimate_learning_time(base_time: int, similarity: float, difficulty: float) → float`

**Scoring Rubric (10-point scale):**
1. Explicit mention frequency
2. Context relevance (project/role context)
3. Temporal recency (recent experience weighted higher)
4. Depth indicators (led, architected, mastered vs. used, familiar)
5. Related skill strength (support skills present)
6. Educational background alignment
7. Certification/credential presence
8. Contribution scope (solo, team, large-scale)
9. Duration and sustained use
10. Demonstrated expertise (publications, open source)

**Formulas:**

**Proficiency Score (No Feedback):**
$$\text{score} = 10 \times (0.80 \times \text{evidence\_strength} + 0.20 \times \text{semantic\_support})$$

**Proficiency Score (With User Feedback):**
$$\text{score} = 0.6 \times \text{score\_model} + 0.4 \times \text{user\_feedback}$$

**Confidence Blending:**
$$\text{confidence} = 0.6 \times \text{llm\_confidence} + 0.25 \times \text{indicator\_density} + 0.15 \times \text{mention\_strength}$$

**Difficulty Estimation:**
$$\text{difficulty} = 10 \times (1 - \text{semantic\_similarity}) - 0.3 \times \text{related\_skill\_strength}$$
$$\text{clamped to } [1, 10]$$

**Time Estimation:**
$$\text{time\_weeks} = \text{base\_time} \times (1 - \text{similarity}) + 0.3 \times \text{base\_time}$$

---

### 4. PathGenerator (`backend/learning_path/path_generator.py`)

**Purpose:** End-to-end DAG construction and prerequisite-aware sequencing

**Key Method:** `generate_learning_path(skills: list[SkillWithScore]) → LearningPath`

**Steps:**
1. **Prerequisite Extraction** (batch LLM call):
   - Single Gemini call for all skills
   - Avoids N×2 rate-limit hits
   - JSON output with markdown fallback

2. **DAG Construction**:
   - Build dependency graph from prerequisites
   - Detect cycles and resolve (remove or warn)
   - Create dependents map (reverse edges)

3. **Priority Scoring**:
   - Formula: `(unlock_power × w_u) / (difficulty^α × time_weeks × w_t)`
   - unlock_power: # of dependent skills in DAG
   - Configurable weights: w_u, w_t, exponent α

4. **Topological Sorting**:
   - DFS with level assignment
   - Within-level priority ranking

5. **Parallel Track Identification**:
   - Skills with no dependencies between them
   - Suggestions for concurrent learning

---

### 5. ResourceAggregator (`backend/skill_resources/main.py`)

**Purpose:** Multi-source learning resource discovery

**Resource Sources:**
- **Wikipedia:** Definitions, overviews, foundational concepts
- **arXiv:** Academic papers, research articles
- **StackExchange:** Technical Q&A, problem-solving
- **Google Books:** Books and textbooks
- **GitHub:** Code repositories and examples
- **YouTube/Web:** Video tutorials and blog posts

**Fallback Strategy:**
1. Query each source in parallel
2. Aggregate and rank results
3. If insufficient results, generate generic search URLs
4. Return best available combination

**Categorization Logic:**
- Tutorial sources: YouTube, blogs, interactive guides
- Documentation: Official docs, wikis, API references
- Courses: Structured course platforms, playlists
- Research: arXiv, academic papers
- Forums: StackExchange, community Q&A
- Books: Google Books, O'Reilly
- Code: GitHub, GitLab, code search

---

### 6. PrerequisiteExtractor (`backend/learning_path/prerequisite_extractor.py`)

**Purpose:** LLM-based batch prerequisite extraction for DAG construction

**Key Method:** `extract_all_prerequisites_batch(skills: list[str]) → dict[str, list[str]]`

**Optimization:**
- Single LLM call for all skills (avoids N separate calls)
- Structured JSON output with parsing fallbacks
- Rate-limit resilience via model fallback chain

**Output:**
```json
{
  "Python": ["Programming Basics", "Algorithms"],
  "Machine Learning": ["Python", "Statistics", "Linear Algebra"],
  "Kubernetes": ["Docker", "Linux", "Networking"],
  ...
}
```

---

### 7. DAGBuilder (`backend/learning_path/dag_builder.py`)

**Purpose:** DAG manipulation, cycle detection, layer extraction, visualization

**Key Methods:**
- `build_dag(prerequisites: dict) → DAG`
- `get_learning_layers() → list[list[str]]`
- `get_topological_sort_by_priority() → list[str]`
- `get_graph_json() → dict`
- `detect_cycles() → list[list[str]]`

**Cycle Detection:** DFS with recursion stack tracking

**Layer Extraction:** BFS-like traversal collecting all skills at each depth level

**Graph JSON Format:**
```json
{
  "nodes": [
    {"id": "skill1", "label": "Skill 1", "level": 0, "priority_score": 8.5},
    ...
  ],
  "edges": [
    {"from": "Python", "to": "Django"},
    ...
  ]
}
```

---

## Algorithms & Formulas

### Algorithm 1: Proficiency Estimation

**Input:** Resume text, skill name

**Process:**
1. Extract resume segments (projects, roles, education)
2. Search for skill mention(s) in segments
3. For each mention, evaluate against 10-indicator rubric
4. Compute score_model as sum of indicators
5. Calculate semantic support (related skills present)
6. Blend evidence and semantic support:
   - $$\text{score} = 10 \times (0.80 \times \text{evidence} + 0.20 \times \text{semantic})$$
7. If user feedback provided: $$\text{score} = 0.6 \times \text{score\_model} + 0.4 \times \text{feedback}$$

**Output:** Proficiency score [0-10], evidence text, confidence [0-1]

---

### Algorithm 2: Priority Scoring

**Input:** Skill name, unlock_power (# of dependents), difficulty, estimated_time_weeks

**Formula:**
$$\text{priority\_score} = \frac{\text{unlock\_power} \times w_u}{(\text{difficulty})^{\alpha} \times \text{time\_weeks} \times w_t}$$

**Parameters:**
- `w_u` = weight for unlock power (default 1.0)
- `w_t` = weight for time/difficulty (default 1.0)
- `α` = exponent for difficulty scaling (default 1.2)

**Interpretation:**
- High unlock_power → high priority (unlocks many dependent skills)
- High difficulty/time → lower priority (harder/longer to learn)
- Configurable weights allow domain-specific tuning

**Example:**
- Docker: unlock_power=5, difficulty=6.5, time=2.5 weeks → score ≈ 8.7
- Advanced orchestration: unlock_power=1, difficulty=8.5, time=6 weeks → score ≈ 2.1

---

### Algorithm 3: Skill Clustering

**Input:** List of extracted skills from JD

**Process:**
1. Embed each skill using sentence-transformer
2. Build cosine similarity matrix
3. Perform hierarchical agglomerative clustering:
   - Linkage: ward (minimizes variance)
   - Distance threshold: 0.38 (configurable)
4. While cluster_count > max_output_skills:
   - Find two closest clusters
   - Merge them
   - Select representative (LLM-based or similarity-based)
5. Return representative skills

**Benefits:**
- Reduces "Kubernetes", "K8s", "container orchestration" → single skill
- Consolidates "ML", "Machine Learning", "Deep Learning" intelligently
- Maintains semantic meaning while reducing noise

---

### Algorithm 4: DAG Construction & Topological Sort

**Input:** Prerequisites mapping (skill → list[prerequisites])

**Process:**
1. Build directed graph from prerequisite edges
2. Detect cycles using DFS (recursion stack tracking)
   - If cycle found, flag for user review
3. Create dependents map (reverse edges)
4. Perform topological sort via DFS:
   - Assign levels (0 = no prerequisites)
   - Within level, sort by priority_score (descending)
5. Extract layers:
   - Level 0: all skills with no prerequisites
   - Level 1: all skills whose prerequisites are in Level 0
   - Continue until all skills assigned

**Output:** Ordered learning sequence with levels

---

### Algorithm 5: Evidence-Based Confidence Estimation

**Input:** LLM confidence (0-1), indicator density (# rubric items met), mention strength (frequency)

**Formula:**
$$\text{confidence} = 0.6 \times \text{llm\_confidence} + 0.25 \times \text{indicator\_density} + 0.15 \times \frac{\text{mentions\_count}}{\text{max\_mentions}}$$

**Rationale:**
- 60% weight on LLM's stated confidence (primary signal)
- 25% weight on rubric coverage (multiple evidence types strengthen)
- 15% weight on mention frequency (repeated mentions increase certainty)

---

## Data Models & Schemas

### Pydantic Models (Skill Analysis)

```python
# Request Models
class AnalyzeSkillsRequest(BaseModel):
    job_description: str
    max_output_skills: int = 20

# Response Models
class SkillScore(BaseModel):
    skill_name: str
    proficiency_score: float  # 0-10
    confidence: float  # 0-1
    evidence: str  # Text context
    similarity: float  # 0-1 (semantic)

class SkillGap(BaseModel):
    skill_name: str
    required: bool
    proficiency: float  # 0-10
    difficulty: float  # 1-10
    estimated_time_weeks: float

class AnalyzeSkillsResponse(BaseModel):
    required_skills: list[str]
    all_skills: list[SkillScore]
    skill_gaps: list[SkillGap]
    gap_count: int
    coverage_percentage: float
```

### Learning Path Models

```python
class SkillWithScore(BaseModel):
    skill_name: str
    priority_score: float
    difficulty: float
    estimated_time_weeks: float
    prerequisites: list[str]
    level: int
    track: str

class LearningPath(BaseModel):
    learning_sequence: list[SkillWithScore]
    learning_layers: list[list[str]]
    parallel_tracks: list[str]
    total_duration_weeks: float
    graph: dict  # nodes and edges for visualization
```

### Resource Models

```python
class Resource(BaseModel):
    title: str
    url: str
    source: str

class ResourceCategory(BaseModel):
    tutorials: list[Resource] = []
    documentation: list[Resource] = []
    courses: list[Resource] = []
    research: list[Resource] = []
    forums: list[Resource] = []
    books: list[Resource] = []
    code: list[Resource] = []

class SkillTransition(BaseModel):
    from_skill: str
    to_skill: str
    resources: ResourceCategory
```

---

## Configuration & Environment Setup

### Environment File (`.env`)

```env
# Gemini API Configuration
GEMINI_API_KEY=your_api_key_here
GEMINI_CHAT_MODEL=gemini-3.1-flash-lite-preview
GEMINI_EMBEDDING_MODEL=text-embedding-004
USE_GEMINI_EMBEDDINGS=false

# Skill Clustering Configuration
MAX_OUTPUT_SKILLS=20
CLUSTER_DISTANCE_THRESHOLD=0.38
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
ALLOW_LLM_CLUSTER_NAMING=true

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Database Configuration (if applicable)
DATABASE_URL=sqlite:///./skillsynapse.db
```

### Configuration Class (`backend/utils/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gemini_api_key: str
    gemini_chat_model: str = "gemini-3.1-flash-lite-preview"
    gemini_embedding_model: str = "text-embedding-004"
    use_gemini_embeddings: bool = False
    
    max_output_skills: int = 20
    cluster_distance_threshold: float = 0.38
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    allow_llm_cluster_naming: bool = True
    
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

---

## Installation & Setup Instructions

### Prerequisites
- Python 3.10+
- pip or conda
- Gemini API key (free tier available)

### Step 1: Clone Repository
```bash
git clone https://github.com/sunjinwoo1298/SkillSynapse.git
cd SkillSynapse
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 4: Configure Environment
```bash
# Create .env file in project root
cat > .env << EOF
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_CHAT_MODEL=gemini-3.1-flash-lite-preview
MAX_OUTPUT_SKILLS=20
CLUSTER_DISTANCE_THRESHOLD=0.38
EOF
```

### Step 5: Download Embedding Models
```bash
# Automatically downloaded on first run by sentence-transformers
# Or manually:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Step 6: Run Development Server
```bash
cd SkillSynapse
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Server running at:** `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs` (Swagger UI)

---

## Usage Examples

### Example 1: Complete Skill Analysis Workflow

**Step 1: Prepare Input Files**
- `job_description.txt`: Target job description
- `resume.pdf`: User's resume

**Step 2: Call Analyze Skills**
```bash
curl -X POST http://localhost:8000/skills/analyze-skills \
  -H "Content-Type: multipart/form-data" \
  -F "job_description=@job_description.txt" \
  -F "resume_file=@resume.pdf" \
  -F "max_output_skills=20"
```

**Step 3: Review Response**
- Identifies 18 required skills from job description
- Scores existing skills: Python (8.2), SQL (7.1), etc.
- Identifies gaps: Kubernetes (0), Docker (1.5), etc.
- Coverage: 71.4%

**Step 4: Provide Feedback (Optional)**
```bash
curl -X POST http://localhost:8000/skills/provide-feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_feedback": {
      "Python": 9.5,
      "Docker": 4.2,
      "Kubernetes": 2.0
    }
  }'
```

---

### Example 2: Generate Learning Path

**Input:**
```json
{
  "skills": [
    {
      "skill_name": "Kubernetes",
      "proficiency": 0,
      "difficulty": 7.1,
      "estimated_time_weeks": 4.2
    },
    {
      "skill_name": "Docker",
      "proficiency": 1.5,
      "difficulty": 6.5,
      "estimated_time_weeks": 2.5
    },
    {
      "skill_name": "Linux Administration",
      "proficiency": 2.0,
      "difficulty": 5.8,
      "estimated_time_weeks": 3.0
    }
  ]
}
```

**Output:**
```json
{
  "learning_sequence": [
    {
      "skill_name": "Linux Administration",
      "priority_score": 9.2,
      "difficulty": 5.8,
      "estimated_time_weeks": 3.0,
      "prerequisites": [],
      "level": 0,
      "track": "infrastructure"
    },
    {
      "skill_name": "Docker",
      "priority_score": 8.7,
      "difficulty": 6.5,
      "estimated_time_weeks": 2.5,
      "prerequisites": ["Linux Administration"],
      "level": 1,
      "track": "containerization"
    },
    {
      "skill_name": "Kubernetes",
      "priority_score": 7.5,
      "difficulty": 7.1,
      "estimated_time_weeks": 4.2,
      "prerequisites": ["Docker"],
      "level": 2,
      "track": "orchestration"
    }
  ],
  "learning_layers": [
    ["Linux Administration"],
    ["Docker"],
    ["Kubernetes"]
  ],
  "parallel_tracks": ["infrastructure", "containerization", "orchestration"],
  "total_duration_weeks": 9.7
}
```

---

### Example 3: Get Learning Resources

**Request:**
```bash
curl -X POST http://localhost:8000/skill-resources/get-resources \
  -H "Content-Type: application/json" \
  -d '{
    "transitions": [
      {"from_skill": "Python", "to_skill": "Data Science"},
      {"from_skill": "SQL", "to_skill": "Big Data"}
    ]
  }'
```

**Response:**
```json
{
  "transitions": [
    {
      "from_skill": "Python",
      "to_skill": "Data Science",
      "resources": {
        "tutorials": [
          {"title": "Python for Data Science", "url": "https://...", "source": "YouTube"},
          {"title": "Pandas Tutorial", "url": "https://...", "source": "Blog"}
        ],
        "documentation": [
          {"title": "NumPy Docs", "url": "https://numpy.org", "source": "Official"}
        ],
        "courses": [
          {"title": "Data Science Specialization", "url": "https://...", "source": "Coursera"}
        ]
      }
    }
  ]
}
```

---

## Deployment Guide

### Docker Deployment

**Step 1: Create Dockerfile**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Build Image**
```bash
docker build -t skillsynapse:latest .
```

**Step 3: Run Container**
```bash
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e MAX_OUTPUT_SKILLS=20 \
  skillsynapse:latest
```

---

### Cloud Deployment (AWS/GCP)

**AWS Lambda + API Gateway:**
1. Zip application: `zip -r deployment.zip backend/ requirements.txt`
2. Upload to Lambda function
3. Configure API Gateway trigger
4. Set environment variables in Lambda console

**Google Cloud Run:**
```bash
gcloud run deploy skillsynapse \
  --source . \
  --runtime python310 \
  --set-env-vars GEMINI_API_KEY=your_key \
  --allow-unauthenticated
```

---

### Production Best Practices

1. **Rate Limiting:**
   - Implement request throttling per user
   - Use Gemini model fallback chain for resilience

2. **Caching:**
   - Cache embedding model in memory
   - Implement Redis for distributed caching

3. **Monitoring:**
   - Log all API requests
   - Track latency per endpoint
   - Monitor Gemini API quota usage

4. **Security:**
   - Validate and sanitize file uploads
   - Use environment variables for secrets
   - Implement API key authentication

5. **Scalability:**
   - Use async/await throughout
   - Consider worker queue for long-running jobs
   - Implement pagination for large results

---

## Troubleshooting & Support

### Common Issues

**Issue 1: "ModuleNotFoundError: No module named 'fitz'"**
```bash
# Solution: Install PyMuPDF
pip install PyMuPDF==1.26.4
```

**Issue 2: "GEMINI_API_KEY not found"**
```bash
# Solution: Create .env file with API key
echo "GEMINI_API_KEY=your_key" > .env
```

**Issue 3: "Slow embedding generation"**
```bash
# Solution: Use faster embedding model
# In .env:
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
```

**Issue 4: "High clustering time for 100+ skills"**
```bash
# Solution: Increase MAX_OUTPUT_SKILLS to filter earlier or use faster embedding
MAX_OUTPUT_SKILLS=30
```

### Performance Optimization

| Bottleneck | Solution |
|---|---|
| LLM API latency | Use lighter model (gemini-flash) or batch operations |
| Embedding generation | Use faster model (all-MiniLM) or GPU acceleration |
| Clustering time | Reduce max_output_skills or increase distance threshold |
| Resume parsing | Limit file size to <10MB; use text extraction libraries |

---

## Future Enhancements

1. **Multi-language Support:** Extend to Hindi, Spanish, Mandarin
2. **User Personalization:** ML-based profile matching for custom recommendations
3. **Real-time Collaboration:** WebSocket support for group learning paths
4. **Mobile App:** React Native frontend for on-the-go access
5. **Enterprise Integration:** LMS integration, SSO, audit logs
6. **Advanced Analytics:** Skill demand forecasting, labor market trends
7. **Certifications:** Auto-discovery and recommendations of relevant certifications
8. **Video Generation:** AI-generated personalized video tutorials

---

## Support & Contact

- **GitHub Issues:** [SkillSynapse Issues](https://github.com/sunjinwoo1298/SkillSynapse/issues)
- **Documentation:** See [README.md](README.md)
- **API Docs:** Visit `/docs` endpoint when server running

---

**End of Documentation**

*Last Updated: March 2026*  
*Version: 1.0*  
*Maintainer: SkillSynapse Team*
