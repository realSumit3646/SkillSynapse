# SkillSynapse

<p align="center">
  <b>From Resume to Role-Ready Roadmap</b><br/>
  AI-powered skill gap analysis and prerequisite-aware learning path generation.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Backend-FastAPI-0ea5e9" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Frontend-React%20%2B%20Vite-22c55e" alt="React" />
  <img src="https://img.shields.io/badge/ML-NLP%20%2B%20Embeddings-f59e0b" alt="ML" />
  <img src="https://img.shields.io/badge/Deploy-Docker%20Compose-6366f1" alt="Docker" />
</p>

## Why SkillSynapse

Career transitions are often inefficient: learners know the destination role but not the optimal route. SkillSynapse turns unstructured job descriptions and resumes into a clear, personalized action plan.

It answers three critical questions in one flow:

1. What skills are required for this role?
2. Which of those skills do I already have?
3. What is the most effective sequence to close my gaps?

## Value Proposition

- Real-world impact: solves a practical, high-frequency problem for students and professionals.
- End-to-end product: analysis, scoring, roadmap generation, and learning resources.
- Intelligent planning: DAG-based sequencing with prerequisite awareness.
- Interactive loop: user feedback updates results without repeating full analysis.
- Demo-ready architecture: API-first backend with a clean React frontend.

## Core Features

| Capability | What It Delivers |
|---|---|
| Skill Extraction | Pulls required skills from job descriptions using AI-assisted NLP |
| Resume Evidence Mapping | Parses PDF/DOCX and identifies demonstrated skills with evidence |
| Skill Gap Detection | Compares required vs detected skills to find learning priorities |
| Proficiency Recalculation | Accepts user feedback and recomputes outcomes from stored state |
| Learning Path Generation | Produces dependency-aware sequencing using graph logic |
| Graph Output | Returns nodes/edges for roadmap visualization in frontend |
| Resource Suggestions | Recommends transition-focused learning resources |

## Product Flow

1. Upload resume and paste target job description.
2. Analyze required skills and current proficiency signals.
3. Identify missing or weak skills.
4. Generate a prerequisite-aware roadmap with phases/tracks.
5. Refine with user feedback and iterate.

## Architecture Snapshot

```text
Client (React + Vite)
        |
        v
FastAPI Routes
  - /analyze-skills
  - /provide-feedback
  - /learning-path/*
        |
        v
Service Layer
  - Skill extraction and clustering
  - Resume parsing and scoring
  - DAG/path generation
  - Resource aggregation
        |
        v
NLP + ML Stack
  - sentence-transformers
  - scikit-learn
  - LangChain integrations
```

## API Highlights

- `POST /analyze-skills`
  Analyzes job description + resume and returns skills, evidence, and gap metrics.

- `POST /provide-feedback`
  Re-runs scoring using user feedback with existing in-memory analysis state.

- `POST /learning-path/generate`
  Builds an optimized learning sequence from skill metadata and prerequisites.

- `POST /learning-path/from-skill-gaps`
  Generates a learning path directly from skill-gap payload data.

- `POST /learning-path/graph`
  Returns graph-ready nodes and edges for frontend roadmap rendering.

## Tech Stack

- Backend: FastAPI, Pydantic, scikit-learn, sentence-transformers, LangChain
- Frontend: React, Vite, modern visualization libraries
- Runtime: Python + Node.js
- Deployment: Docker Compose

## Quick Start (Docker)

```bash
git clone https://github.com/sunjinwoo1298/SkillSynapse.git
cd SkillSynapse
docker compose up --build
```

Services:

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Local Development

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend:

```bash
cd client
npm install
npm run dev -- --port 3000
```

## Repository Structure

- `backend/` API routes, NLP/ML services, learning-path engine.
- `client/` React application for evaluation and roadmap UI.
- `DOCUMENTATION.md` deeper technical and implementation notes.

## Configuration

- AI-assisted modules may require environment variables/API keys.
- CORS is currently open for development convenience.

## Demo Pitch (One Line)

SkillSynapse is the GPS for career transitions: it diagnoses skill gaps and gives users the fastest, smartest route to their target role.

## License

Add license information (for example, MIT) before public release.
