"""
FastAPI backend for fetching transition resources between two skills.

Sources: Wikipedia API, arXiv API, StackExchange API, Google Books API,
GitHub API (or Gemini fallback), and Gemini-curated YouTube / website resources.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


def _load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


# Try several candidate locations so this works both standalone and mounted
# as a sub-app (where CWD is the repo root, not backend/).
for _env_candidate in ("backend/.env", ".env", "../.env"):
    _load_env_file(_env_candidate)


REQUEST_TIMEOUT_SECONDS = 8
HTTP_HEADERS = {
    "User-Agent": "SkillSynapse-FreeResourceFetcher/1.0",
    "Accept": "application/json, application/atom+xml;q=0.9, */*;q=0.8",
}


class ResourceItem(BaseModel):
    title: str
    url: str
    source: str
    level: str
    relevance_score: float
    image_url: str | None = None


class TransitionRequest(BaseModel):
    from_skill: str = Field(alias="from", min_length=1)
    to_skill: str = Field(alias="to", min_length=1)

    class Config:
        populate_by_name = True


class TransitionResources(BaseModel):
    from_skill: str = Field(serialization_alias="from")
    to_skill: str = Field(serialization_alias="to")
    resources: Dict[str, List[ResourceItem]]


class TransitionResponse(BaseModel):
    result: TransitionResources


def _safe_json_get(url: str, headers: Dict[str, str] | None = None) -> Dict:
    req = urllib.request.Request(url, headers=headers or HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    if not payload.strip():
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {}


def _safe_text_get(url: str, headers: Dict[str, str] | None = None) -> str:
    req = urllib.request.Request(url, headers=headers or HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="ignore")


def _level_from_query(query: str) -> str:
    q = query.lower()
    if "roadmap" in q or "prerequisite" in q or "introduction" in q or "survey" in q:
        return "beginner"
    if "advanced" in q or "architecture" in q:
        return "advanced"
    return "intermediate"


def _relevance(query: str, title: str) -> float:
    """Token overlap relevance — deterministic, no random noise."""
    q_tokens = {t for t in re.split(r"\W+", query.lower()) if len(t) > 2}
    t_tokens = {t for t in re.split(r"\W+", title.lower()) if len(t) > 2}
    if not q_tokens:
        return 0.60
    overlap = len(q_tokens & t_tokens)
    return round(min(0.55 + overlap / len(q_tokens) * 0.40, 0.97), 3)


def _to_resource_items(raw: List[Dict], query: str, source: str) -> List[ResourceItem]:
    level = _level_from_query(query)
    out: List[ResourceItem] = []
    for row in raw[:3]:
        title = row.get("title", "Untitled Resource")
        url = row.get("url", "https://example.com")
        image_url = row.get("image_url")
        out.append(
            ResourceItem(
                title=title,
                url=url,
                source=source,
                level=level,
                relevance_score=_relevance(query, title),
                image_url=image_url,
            )
        )
    return out


def fetch_wikipedia(query: str, limit: int = 3) -> List[Dict]:
    encoded = urllib.parse.quote(query)
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=opensearch&search={encoded}&limit={limit}&namespace=0&format=json"
    )
    data = _safe_json_get(url)
    titles = data[1] if len(data) > 1 else []
    links = data[3] if len(data) > 3 else []

    results = []
    for idx, title in enumerate(titles[:limit]):
        link = links[idx] if idx < len(links) else f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        results.append({"title": title, "url": link})
    return results


def fetch_arxiv(query: str, limit: int = 3) -> List[Dict]:
    encoded = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results={limit}"
    xml_payload = _safe_text_get(url)
    if not xml_payload.strip():
        return []

    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    results: List[Dict] = []
    for entry in entries[:limit]:
        title_node = entry.find("atom:title", ns)
        link_node = entry.find("atom:id", ns)
        title = title_node.text.strip() if title_node is not None and title_node.text else "arXiv paper"
        link = link_node.text.strip() if link_node is not None and link_node.text else "https://arxiv.org"
        results.append({"title": title, "url": link})
    return results


def fetch_stackexchange(query: str, limit: int = 3) -> List[Dict]:
    encoded = urllib.parse.quote(query)
    url = (
        "https://api.stackexchange.com/2.3/search/advanced"
        f"?order=desc&sort=votes&q={encoded}&site=stackoverflow&pagesize={limit}"
    )
    data = _safe_json_get(url)
    if not data:
        return []

    results = []
    for item in data.get("items", [])[:limit]:
        title = html.unescape(item.get("title", "Stack Overflow discussion"))
        link = item.get("link", "https://stackoverflow.com")
        owner = item.get("owner", {}) if isinstance(item.get("owner", {}), dict) else {}
        results.append({
            "title": title,
            "url": link,
            "image_url": owner.get("profile_image"),
        })
    return results


def fetch_google_books(query: str, limit: int = 3) -> List[Dict]:
    encoded = urllib.parse.quote(query)
    url = f"https://www.googleapis.com/books/v1/volumes?q={encoded}&maxResults={limit}"
    data = _safe_json_get(url)
    if not data:
        return []

    results = []
    for item in data.get("items", [])[:limit]:
        info = item.get("volumeInfo", {})
        title = info.get("title", "Book")
        link = info.get("previewLink") or info.get("infoLink") or "https://books.google.com"
        image_links = info.get("imageLinks", {}) if isinstance(info.get("imageLinks", {}), dict) else {}
        image_url = image_links.get("thumbnail") or image_links.get("smallThumbnail")
        results.append({"title": title, "url": link, "image_url": image_url})
    return results


def fetch_github(query: str, limit: int = 3) -> List[Dict]:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return []  # Curator provides GitHub recommendations when no API token is available

    encoded = urllib.parse.quote(query)
    url = f"https://api.github.com/search/repositories?q={encoded}&sort=stars&order=desc&per_page={limit}"
    headers = {
        **HTTP_HEADERS,
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = _safe_json_get(url, headers)
    if not data:
        return []

    results = []
    for repo in data.get("items", [])[:limit]:
        full_name = repo.get("full_name", "repository")
        description = repo.get("description") or "GitHub repository"
        link = repo.get("html_url", "https://github.com")
        owner = repo.get("owner", {}) if isinstance(repo.get("owner", {}), dict) else {}
        results.append({
            "title": f"{full_name} — {description}",
            "url": link,
            "image_url": owner.get("avatar_url"),
        })
    return results


# ---------------------------------------------------------------------------
# Gemini curator — produces specific named resources instead of search links
# ---------------------------------------------------------------------------

_CURATOR_PROMPT = """\
You are a senior technical educator curating learning resources for a skill transition.

Skill transition: "{from_skill}" -> "{to_skill}"

Return ONLY a valid JSON object — no markdown fences, no explanation, nothing else.

{{
  "youtube": [
    {{"title": "exact video or playlist title", "url": "https://www.youtube.com/watch?v=XXXX"}},
    {{"title": "exact video or playlist title", "url": "https://www.youtube.com/playlist?list=XXXX"}}
  ],
  "websites": [
    {{"title": "exact course or tutorial title", "url": "https://real-url.com/path"}},
    {{"title": "exact course or tutorial title", "url": "https://real-url.com/path"}}
  ],
  "github": [
    {{"title": "owner/repo - short description", "url": "https://github.com/owner/repo"}},
    {{"title": "owner/repo - short description", "url": "https://github.com/owner/repo"}}
  ],
  "books": [
    {{"title": "Book Title by Author", "url": "https://real-url.com"}},
    {{"title": "Book Title by Author", "url": "https://real-url.com"}}
  ]
}}

Rules (strictly follow):
- 2 items per category.
- Every URL must be a direct link to the real resource, never a search page.
- YouTube: use channels like freeCodeCamp, MIT OCW, 3Blue1Brown, StatQuest, Sentdex, Fireship, Traversy Media.
- Websites: use fast.ai, Kaggle Learn, Real Python, official docs, roadmap.sh, Coursera, edX.
- GitHub: only well-known repos with thousands of stars (e.g. awesome-* lists, official repos).
- Books: classic textbooks. Use the publisher page, O'Reilly, or a well-known legitimate URL.
"""


def _extract_json(text: str) -> str:
    """Pull the first complete JSON object out of text, stripping any markdown fences."""
    # Remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```", "", text)
    # Find outermost { }
    start = text.find("{")
    end = text.rfind("}") + 1
    if 0 <= start < end:
        return text[start:end]
    return text.strip()


class GeminiResourceCurator:
    """Use Gemini to produce specific, named learning resources (not search links)."""

    def __init__(self) -> None:
        self._api_key: str | None = os.getenv("GEMINI_API_KEY")

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    async def curate(self, from_skill: str, to_skill: str) -> Dict:
        if not self.enabled:
            print("WARNING [curator]: GEMINI_API_KEY not set — YouTube/Websites/GitHub/Books will be empty.")
            return {}
        try:
            import google.generativeai as genai  # always installed via google-generativeai

            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={"temperature": 0.15, "response_mime_type": "application/json"},
            )
            prompt = _CURATOR_PROMPT.format(from_skill=from_skill, to_skill=to_skill)
            response = await asyncio.to_thread(model.generate_content, prompt)
            content = _extract_json(response.text.strip())
            data = json.loads(content)
            print(f"INFO [curator]: OK — got {sum(len(v) for v in data.values() if isinstance(v, list))} resources for {from_skill}→{to_skill}")
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            print(f"WARNING [curator]: Gemini call failed ({type(exc).__name__}): {exc}")
            return {}

    @staticmethod
    def _is_search_url(url: str) -> bool:
        search_patterns = (
            "youtube.com/results",
            "youtube.com/search",
            "google.com/search",
            "duckduckgo.com",
            "bing.com/search",
            "github.com/search",
            "stackoverflow.com/search",
            "wikipedia.org/w/index.php?search",
        )
        return any(pat in url for pat in search_patterns)

    def _to_raw(self, items: List[Dict]) -> List[Dict]:
        results: List[Dict] = []
        for item in items or []:
            url = item.get("url", "")
            title = item.get("title", "")
            if not url or not title or self._is_search_url(url):
                continue
            if not url.startswith(("http://", "https://")):
                continue
            results.append({"title": title, "url": url})
        return results


_CURATOR = GeminiResourceCurator()


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def _safe_fetch(func, *args, **kwargs) -> List[Dict]:
    """Run a blocking fetch in a thread pool; return [] on any error."""
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except Exception:
        return []


async def fetch_transition_resources(from_skill: str, to_skill: str) -> Dict[str, List[ResourceItem]]:
    arxiv_query  = f"{to_skill} {from_skill} introduction survey"
    books_query  = f"learn {to_skill} {from_skill} programming"
    github_query = f"{to_skill} {from_skill} awesome"
    wiki_query   = to_skill
    stack_query  = f"{to_skill} {from_skill} how to"

    (
        curator_result,
        papers_raw,
        books_raw,
        github_raw,
        wiki_raw,
        stack_raw,
    ) = await asyncio.gather(
        _CURATOR.curate(from_skill, to_skill),
        _safe_fetch(fetch_arxiv, arxiv_query, limit=3),
        _safe_fetch(fetch_google_books, books_query, limit=3),
        _safe_fetch(fetch_github, github_query, limit=3),
        _safe_fetch(fetch_wikipedia, wiki_query, limit=3),
        _safe_fetch(fetch_stackexchange, stack_query, limit=3),
    )

    yt_raw    = _CURATOR._to_raw(curator_result.get("youtube", []))
    web_raw   = _CURATOR._to_raw(curator_result.get("websites", []))
    cur_books = _CURATOR._to_raw(curator_result.get("books", []))
    cur_gh    = _CURATOR._to_raw(curator_result.get("github", []))

    # Prefer API results where available; fall back to curator
    if not github_raw:
        github_raw = cur_gh
    if not books_raw:
        books_raw = cur_books

    resources = {
        "research_papers": _to_resource_items(papers_raw, arxiv_query, "arXiv"),
        "books":           _to_resource_items(books_raw, books_query, "Google Books / Gemini"),
        "github":          _to_resource_items(github_raw, github_query, "GitHub"),
        "youtube":         _to_resource_items(yt_raw, f"learn {to_skill}", "YouTube"),
        "websites":        _to_resource_items(web_raw, f"{to_skill} tutorial", "Web"),
        "documentation":   _to_resource_items(wiki_raw + stack_raw, f"{to_skill} documentation", "Wikipedia / StackOverflow"),
    }

    for category, items in resources.items():
        if items:
            continue
        q = urllib.parse.quote_plus(f"{to_skill} {from_skill} {category.replace('_', ' ')}")
        resources[category] = [
            ResourceItem(
                title=f"Search: {to_skill} — {category.replace('_', ' ')}",
                url=f"https://duckduckgo.com/?q={q}",
                source="Fallback",
                level="intermediate",
                relevance_score=0.55,
                image_url=None,
            )
        ]

    return resources


# ---------------------------------------------------------------------------
# App / router
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Skill Transition Resource Fetcher",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(tags=["skill-resources"])


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "skill-transition-resource-fetcher"}


@router.post("/get-resources", response_model=TransitionResponse)
async def get_resources(request: TransitionRequest) -> TransitionResponse:
    from_skill = request.from_skill.strip()
    to_skill = request.to_skill.strip()
    if not from_skill or not to_skill:
        raise HTTPException(status_code=400, detail="Both 'from' and 'to' must be non-empty strings.")

    resources = await fetch_transition_resources(from_skill, to_skill)
    return TransitionResponse(
        result=TransitionResources(
            from_skill=from_skill,
            to_skill=to_skill,
            resources=resources,
        )
    )


app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
