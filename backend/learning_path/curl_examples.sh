#!/bin/bash
# Learning Path API Examples with DAG
# Test these endpoints with: bash backend/learning_path/curl_examples.sh

API_URL="http://localhost:8000"

echo "=== Learning Path API Examples (DAG-Based) ==="
echo ""

# Example 1: Generate Complete Learning Path with DAG
echo "1. Generate Learning Path with DAG (Full Example)"
echo "POST /learning-path/generate"
curl -X POST "$API_URL/learning-path/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "skills_to_learn": ["Docker", "Kubernetes", "AWS", "Terraform"],
    "skill_metadata": {
      "Docker": {
        "difficulty": 6,
        "time": "2 weeks",
        "unlock_power": 8,
        "prerequisites": [],
        "description": "Containerization platform"
      },
      "Kubernetes": {
        "difficulty": 9,
        "time": "2-3 months",
        "unlock_power": 4,
        "prerequisites": ["Docker"],
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
    "auto_extract_prerequisites": false,
    "available_time_weeks": 16,
    "max_difficulty_per_phase": 8
  }' | jq .

echo ""
echo "----------------------------------------"
echo ""

# Example 2: Generate Path with LLM Prerequisite Extraction
echo "2. Generate Learning Path with Auto-Extracted Prerequisites (LLM)"
echo "POST /learning-path/generate"
curl -X POST "$API_URL/learning-path/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "skills_to_learn": ["Docker", "Kubernetes", "Python"],
    "skill_metadata": {
      "Docker": {
        "difficulty": 6,
        "time": "2 weeks",
        "unlock_power": 8
      },
      "Kubernetes": {
        "difficulty": 9,
        "time": "2-3 months",
        "unlock_power": 4
      },
      "Python": {
        "difficulty": 2,
        "time": "3 days",
        "unlock_power": 10
      }
    },
    "auto_extract_prerequisites": true
  }' | jq .

echo ""
echo "----------------------------------------"
echo ""

# Example 3: Score Skills Only
echo "3. Score Skills (Without DAG Generation)"
echo "POST /learning-path/score-skills"
curl -X POST "$API_URL/learning-path/score-skills" \
  -H "Content-Type: application/json" \
  -d '{
    "Docker": {
      "difficulty": 6,
      "time": "2 weeks",
      "unlock_power": 8
    },
    "Kubernetes": {
      "difficulty": 9,
      "time": "2-3 months",
      "unlock_power": 4
    },
    "Python": {
      "difficulty": 2,
      "time": "3 days",
      "unlock_power": 10
    }
  }' | jq .

echo ""
echo "----------------------------------------"
echo ""

# Example 4: Visualize DAG
echo "4. Visualize DAG Structure"
echo "POST /learning-path/dag-visualize"
curl -X POST "$API_URL/learning-path/dag-visualize" \
  -H "Content-Type: application/json" \
  -d '{
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
      },
      "AWS": {
        "difficulty": 7,
        "time": "3-4 weeks",
        "unlock_power": 7,
        "prerequisites": ["Docker"]
      }
    },
    "auto_extract_prerequisites": false
  }' | jq .

echo ""
echo "----------------------------------------"
echo ""

# Example 5: Get Configuration
echo "5. Get Current Configuration"
echo "GET /learning-path/config"
curl -X GET "$API_URL/learning-path/config" | jq .

echo ""
echo "----------------------------------------"
echo ""

# Example 6: BATCH LLM Extraction (Optimized - No Prerequisites Provided)
echo "6. Batch LLM Extraction (auto_extract_prerequisites = true)"
echo "POST /learning-path/generate"
echo "NOTE: No prerequisites provided - LLM extracts ALL in single request"
curl -X POST "$API_URL/learning-path/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "skills_to_learn": ["Docker", "Kubernetes", "AWS", "Terraform", "CI/CD"],
    "skill_metadata": {
      "Docker": {
        "difficulty": 6,
        "time": "2 weeks",
        "unlock_power": 8
      },
      "Kubernetes": {
        "difficulty": 9,
        "time": "2-3 months",
        "unlock_power": 4
      },
      "AWS": {
        "difficulty": 7,
        "time": "3-4 weeks",
        "unlock_power": 7
      },
      "Terraform": {
        "difficulty": 5,
        "time": "1 week",
        "unlock_power": 6
      },
      "CI/CD": {
        "difficulty": 4,
        "time": "2 weeks",
        "unlock_power": 5
      }
    },
    "auto_extract_prerequisites": true
  }' | jq .

echo ""
echo "----------------------------------------"
echo ""

# Example 7: Minimal Example (Just Required Fields)
echo "7. Minimal Example (Only Required Fields)"
echo "POST /learning-path/generate"
curl -X POST "$API_URL/learning-path/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "skills_to_learn": ["Docker", "Kubernetes"],
    "skill_metadata": {
      "Docker": {
        "difficulty": 6,
        "time": "2 weeks",
        "unlock_power": 8
      },
      "Kubernetes": {
        "difficulty": 9,
        "time": "2-3 months",
        "unlock_power": 4,
        "prerequisites": ["Docker"]
      }
    }
  }' | jq .

echo ""
echo "----------------------------------------"
echo ""

# Example 8: Get Lightweight Graph for Visualization
echo "8. Get Graph Structure for Visualization (D3.js, Cytoscape, etc.)"
echo "POST /learning-path/graph"
curl -X POST "$API_URL/learning-path/graph" \
  -H "Content-Type: application/json" \
  -d '{
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
    "auto_extract_prerequisites": false
  }' | jq .

echo ""
echo "========================================"
echo "All examples completed!"
