"""LLM-based Prerequisite Extractor using Gemini - Batch Processing."""

import json
import re
from typing import List, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.utils.config import settings


class PrerequisiteExtractor:
    """
    Uses LLM (Google Gemini) to extract prerequisites for ALL skills in a single request.
    
    Optimization:
    - Single API call for all skills (avoids rate limiting)
    - Batch processing is more efficient
    - Returns prerequisite map for entire skill set
    
    Example:
        Input: ["Docker", "Kubernetes", "AWS"]
        Single LLM call returns: {
            "Docker": [],
            "Kubernetes": ["Docker"],
            "AWS": ["Docker"]
        }
    """
    
    def __init__(self) -> None:
        """Initialize with LLM model."""
        self.chat_models = self._build_chat_models(settings.gemini_chat_model)
        self.chat_model_idx = 0
        
        # Single prompt for batch prerequisite extraction
        self.batch_prerequisite_prompt = PromptTemplate.from_template(
            """
You are an expert in skill learning paths and technical education.

For EACH skill in the list below, identify ONLY the hard prerequisites (skills that MUST be learned first).

Rules:
1. Only include skills from the provided list as prerequisites
2. Return as valid JSON only (no other text)
3. If a skill has no prerequisites, use empty array []
4. Be conservative - only include what's truly necessary

Skills to analyze: {skills_list}

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
    "skill_name_1": ["prerequisite1", "prerequisite2"],
    "skill_name_2": [],
    "skill_name_3": ["prerequisite1"]
}}

Example output:
{{
    "Docker": [],
    "Kubernetes": ["Docker"],
    "Python": []
}}
""".strip()
        )
        
        self.chain: RunnableSequence | None = None
        if settings.gemini_api_key and self.chat_models:
            self.chain = self._build_chain(self.chat_models[self.chat_model_idx])
    
    async def extract_all_prerequisites_batch(
        self,
        skills: List[str]
    ) -> Dict[str, List[str]]:
        """
        Extract prerequisites for ALL skills in a SINGLE LLM call.
        
        This is the optimized approach that:
        - Makes only ONE API call (no rate limiting issues)
        - Gets prerequisites for all skills at once
        - Batch processing is more efficient
        
        Args:
            skills: List of all skills to analyze
            
        Returns:
            {skill_name: [prerequisite_names]} for ALL skills
            
        Raises:
            RuntimeError: If GEMINI_API_KEY not set or LLM fails
        """
        if self.chain is None:
            raise RuntimeError("GEMINI_API_KEY is required for LLM prerequisite extraction")
        
        if not skills:
            return {}
        
        # Single request to LLM for all skills
        while True:
            try:
                skills_list = "\n".join(f"- {skill}" for skill in skills)
                raw = await self.chain.ainvoke({
                    "skills_list": skills_list
                })
                break
            except Exception as exc:
                if self._is_rate_limit_error(exc) and self.chat_model_idx + 1 < len(self.chat_models):
                    # Fallback to next model on rate limit
                    self.chat_model_idx += 1
                    self.chain = self._build_chain(self.chat_models[self.chat_model_idx])
                    continue
                raise
        
        # Parse JSON response
        prerequisites_map = self._parse_json_response(raw.strip(), skills)
        
        return prerequisites_map
    
    @staticmethod
    def _parse_json_response(raw: str, skills: List[str]) -> Dict[str, List[str]]:
        """
        Parse JSON response from LLM.
        
        Handles:
        - JSON wrapped in markdown code blocks
        - JSON with extra explanatory text
        - Incomplete or malformed JSON
        
        Args:
            raw: Raw LLM response
            skills: Expected skill names (for validation)
            
        Returns:
            Clean prerequisites map {skill: [prereqs]}
        """
        skills_set = set(skills)
        
        # Try direct JSON parse
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return PrerequisiteExtractor._validate_prerequisites_map(
                    parsed, skills_set
                )
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from markdown code block
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw)
        if code_block_match:
            try:
                parsed = json.loads(code_block_match.group(1))
                if isinstance(parsed, dict):
                    return PrerequisiteExtractor._validate_prerequisites_map(
                        parsed, skills_set
                    )
            except json.JSONDecodeError:
                pass
        
        # Try extracting raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict):
                    return PrerequisiteExtractor._validate_prerequisites_map(
                        parsed, skills_set
                    )
            except json.JSONDecodeError:
                pass
        
        # Fallback: return empty prerequisite map
        print(f"Warning: Could not parse LLM response. Raw response: {raw[:200]}")
        return {skill: [] for skill in skills}
    
    @staticmethod
    def _validate_prerequisites_map(
        parsed: dict,
        valid_skills: set
    ) -> Dict[str, List[str]]:
        """
        Validate and clean prerequisites map.
        
        Ensures:
        - All prerequisites are valid skill names
        - All skills have prerequisite lists
        - No invalid entries
        """
        result = {}
        
        for skill, prereqs in parsed.items():
            skill_str = str(skill).strip()
            
            # Ensure prereqs is a list
            if not isinstance(prereqs, list):
                prereqs = []
            
            # Filter to only valid skills
            valid_prereqs = [
                str(p).strip()
                for p in prereqs
                if str(p).strip() in valid_skills
            ]
            
            result[skill_str] = valid_prereqs
        
        return result
    
    def _build_chain(self, model_name: str) -> RunnableSequence:
        """Build LangChain pipeline for batch prerequisite extraction."""
        llm = ChatGoogleGenerativeAI(
            google_api_key=settings.gemini_api_key,
            model=model_name.removeprefix("models/"),
            temperature=0,  # Deterministic output
        )
        return self.batch_prerequisite_prompt | llm | StrOutputParser()
    
    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Check if exception is a rate limit error."""
        err = str(exc).lower()
        return "429" in err or "resource_exhausted" in err or "rate" in err
    
    @staticmethod
    def _build_chat_models(primary: str) -> List[str]:
        """Build list of fallback models for rate limit resilience."""
        candidates = [
            primary,
            "models/gemini-2.5-flash",
            "models/gemini-2.5-flash-lite",
            "models/gemini-flash-latest",
            "models/gemini-flash-lite-latest",
            "models/gemini-2.5-pro",
            "models/gemini-pro-latest",
            "models/gemini-3-flash-preview",
            "models/gemini-3.1-flash-lite-preview",
        ]
        
        models: List[str] = []
        seen = set()
        for model in candidates:
            key = model.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            models.append(key)
        return models

