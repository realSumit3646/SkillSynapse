import re
from typing import Iterable


ACRONYMS = {
    "api",
    "aws",
    "gcp",
    "ci",
    "cd",
    "sql",
    "nosql",
    "nlp",
    "ml",
    "ai",
    "llm",
    "oop",
    "ui",
    "ux",
    "rest",
    "graphql",
    "json",
    "xml",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9#+./\- ]+", " ", text)
    return normalize_whitespace(text)


def display_name(skill: str) -> str:
    tokens = normalize_whitespace(skill).split(" ")
    out = []
    for tok in tokens:
        base = tok.strip()
        low = base.lower()
        if low in ACRONYMS:
            out.append(low.upper())
        elif any(ch.isupper() for ch in base):
            out.append(base)
        else:
            out.append(base.capitalize())
    return " ".join(out)


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        key = normalize_for_match(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(normalize_whitespace(item))
    return result


def is_explicit_in_text(skill: str, source_text: str) -> bool:
    skill_norm = normalize_for_match(skill)
    text_norm = normalize_for_match(source_text)
    if not skill_norm or not text_norm:
        return False

    if skill_norm in text_norm:
        return True

    skill_tokens = [t for t in skill_norm.split(" ") if t]
    if not skill_tokens:
        return False

    pattern = r"\\b" + r"\\s+".join(re.escape(token) for token in skill_tokens) + r"\\b"
    return re.search(pattern, text_norm) is not None
