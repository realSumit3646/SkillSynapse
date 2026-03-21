from __future__ import annotations

import math
import re
from collections import deque

WHITESPACE_PATTERN = re.compile(r"\s+")
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

NEAR_STRONG = ("project", "built", "developed")
NEAR_WEAK = ("internship", "experience")

UNLOCK_GRAPH: dict[str, list[str]] = {
    "python": ["numpy", "pandas", "ml"],
    "ml": ["deep learning", "nlp"],
    "docker": ["kubernetes", "ci/cd"],
}


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text.lower()).strip()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def classify_base_time_days(skill: str) -> float:
    s = skill.lower()
    if any(token in s for token in ["language", "programming", "system design"]):
        return 75.0
    if any(token in s for token in ["framework", "tool", "cloud"]):
        return 30.0
    if any(token in s for token in ["library", "api"]):
        return 9.0
    return 30.0


def format_time(days: float) -> str:
    d = max(1, int(round(days)))
    if d <= 7:
        return f"{d} days"
    if d <= 30:
        weeks = max(1, int(round(d / 7)))
        return f"{weeks} weeks"

    months = d / 30.0
    lo = max(1, int(math.floor(months)))
    hi = max(lo + 1, int(math.ceil(months)))
    return f"{lo}-{hi} months"


def compute_unlock_power(skill: str, graph: dict[str, list[str]] | None = None) -> int:
    graph = graph or UNLOCK_GRAPH
    start = skill.lower()
    visited: set[str] = set()
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for nxt in graph.get(current, []):
            nxt_l = nxt.lower()
            if nxt_l not in visited:
                visited.add(nxt_l)
                queue.append(nxt_l)

    return len(visited)
