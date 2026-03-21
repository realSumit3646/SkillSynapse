from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.utils.config import settings
from backend.utils.text_utils import dedupe_preserve_order, display_name, is_explicit_in_text


class SkillExtractorService:
    def __init__(self) -> None:
        self.chat_models = self._build_chat_models(settings.gemini_chat_model)
        self.chat_model_idx = 0

        self.prompt = PromptTemplate.from_template(
            """
    Task: extract skills explicitly present in the text.
    Rules:
    - No inference or guessing.
    - Keep names short.
    - Include only explicit languages, frameworks, libraries, tools, platforms, and soft skills.
    - Return JSON array of strings only.

Text:
{text}
""".strip()
        )

        self.chain: RunnableSequence | None = None
        if settings.gemini_api_key and self.chat_models:
            self.chain = self._build_chain(self.chat_models[self.chat_model_idx])

    async def extract_skills(self, text: str) -> list[str]:
        if self.chain is None:
            raise RuntimeError("GEMINI_API_KEY is required for LangChain skill extraction")

        raw = ""
        while True:
            try:
                raw = await self.chain.ainvoke({"text": text})
                break
            except Exception as exc:
                if self._is_rate_limit_error(exc) and self.chat_model_idx + 1 < len(self.chat_models):
                    self.chat_model_idx += 1
                    self.chain = self._build_chain(self.chat_models[self.chat_model_idx])
                    continue
                raise

        candidates = self._parse_output(raw)

        strict = [skill for skill in candidates if is_explicit_in_text(skill, text)]
        strict = dedupe_preserve_order(strict)
        return [display_name(skill) for skill in strict]

    @staticmethod
    def _parse_output(raw: str) -> list[str]:
        raw = raw.strip()
        parsed = SkillExtractorService._parse_json_array(raw)
        if parsed is not None:
            return parsed

        json_match = re.search(r"\[[\s\S]*\]", raw)
        if json_match:
            parsed = SkillExtractorService._parse_json_array(json_match.group(0))
            if parsed is not None:
                return parsed

        lines = []
        for line in raw.splitlines():
            item = re.sub(r"^[-*\d.\)\s]+", "", line).strip()
            if item:
                lines.append(item)
        return lines

    @staticmethod
    def _parse_json_array(value: str) -> list[str] | None:
        try:
            parsed: Any = json.loads(value)
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, list):
            return None

        result = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
        return result

    def _build_chain(self, model_name: str) -> RunnableSequence:
        llm = ChatGoogleGenerativeAI(
            google_api_key=settings.gemini_api_key,
            model=model_name.removeprefix("models/"),
            temperature=0,
        )
        return self.prompt | llm | StrOutputParser()

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        err = str(exc).lower()
        return "429" in err or "resource_exhausted" in err or "rate" in err

    @staticmethod
    def _build_chat_models(primary: str) -> list[str]:
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

        models: list[str] = []
        seen = set()
        for model in candidates:
            key = model.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            models.append(key)
        return models
