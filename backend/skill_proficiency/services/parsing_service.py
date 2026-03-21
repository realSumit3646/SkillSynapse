from __future__ import annotations

import ast
import json
from typing import Any

import fitz
from docx import Document
from fastapi import HTTPException, UploadFile, status

from utils.common import clamp


def _strip_wrapping_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def _parse_json_or_python_literal(raw: str) -> Any:
    text = _strip_wrapping_quotes(raw)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def extract_text_from_pdf(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n".join(pages)


def extract_text_from_docx(data: bytes) -> str:
    import io

    document = Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)


def extract_resume_text(upload: UploadFile, file_bytes: bytes) -> str:
    filename = (upload.filename or "").lower()
    content_type = (upload.content_type or "").lower()

    try:
        if filename.endswith(".pdf") or "pdf" in content_type:
            return extract_text_from_pdf(file_bytes)
        if filename.endswith(".docx") or "wordprocessingml" in content_type or "docx" in content_type:
            return extract_text_from_docx(file_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read resume file: {exc}",
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unsupported file type. Please upload a PDF or DOCX file.",
    )


def parse_skills(skills_raw: str) -> list[str]:
    raw = (skills_raw or "").strip()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="skills is required.",
        )

    parsed = _parse_json_or_python_literal(raw)

    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        cleaned = [s.strip() for s in parsed if s.strip()]
    else:
        # Fallback for plain comma-separated text from form fields.
        cleaned = [token.strip() for token in _strip_wrapping_quotes(raw).split(",") if token.strip()]

    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="skills must be a JSON list of strings or comma-separated skill names.",
        )
    return cleaned


def parse_feedback(feedback_raw: str | None) -> dict[str, float]:
    if not feedback_raw:
        return {}

    parsed = _parse_json_or_python_literal(feedback_raw)

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="user_feedback must be a JSON object {skill: rating}.",
        )

    normalized: dict[str, float] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, (int, float)):
            continue
        normalized[key.lower().strip()] = clamp(float(value), 0.0, 10.0)
    return normalized
