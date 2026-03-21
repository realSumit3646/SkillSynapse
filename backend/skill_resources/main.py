"""
FastAPI backend for fetching transition resources between two skills.

Input is only two strings: from -> to.
Sources: Wikipedia API, arXiv API, StackExchange API, Google Books API,
GitHub search API (or fallback links), and YouTube search links.
"""

from __future__ import annotations

import html
import json
import os
import random
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

def _load_env_file(path: str = ".env") -> None:
	"""Minimal .env loader so token import works without extra dependencies."""
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
		# If parsing fails, continue with normal environment variables.
		pass


_load_env_file()


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
	edge: str
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
	if "roadmap" in q or "prerequisite" in q:
		return "beginner"
	if "advanced" in q or "architecture" in q:
		return "advanced"
	return "intermediate"


def _relevance(query: str, title: str) -> float:
	q_tokens = {t for t in query.lower().split() if t}
	t_tokens = {t for t in title.lower().split() if t}
	overlap = len(q_tokens.intersection(t_tokens))
	base = 0.55 + min(overlap * 0.08, 0.35)
	return round(min(base + random.uniform(0.0, 0.08), 0.98), 3)


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


def _fetch_wikipedia_image(title: str) -> str | None:
	"""Fetch thumbnail URL for a Wikipedia page title when available."""
	safe_title = urllib.parse.quote(title.replace(" ", "_"))
	url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}"
	data = _safe_json_get(url)
	thumb = data.get("thumbnail", {}) if isinstance(data, dict) else {}
	image_url = thumb.get("source") if isinstance(thumb, dict) else None
	return image_url


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
		results.append({
			"title": title,
			"url": link,
			"image_url": _fetch_wikipedia_image(title),
		})
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
		results.append({
			"title": title,
			"url": link,
			"image_url": image_url,
		})
	return results


def fetch_github(query: str, limit: int = 3) -> List[Dict]:
	token = os.getenv("GITHUB_TOKEN")
	if not token:
		q = urllib.parse.quote_plus(query)
		return [
			{"title": f"GitHub repositories: {query}", "url": f"https://github.com/search?q={q}&type=repositories", "image_url": None},
			{"title": f"GitHub code search: {query}", "url": f"https://github.com/search?q={q}&type=code", "image_url": None},
			{"title": f"GitHub topics: {query}", "url": f"https://github.com/topics/{urllib.parse.quote(query.lower().replace(' ', '-'))}", "image_url": None},
		][:limit]

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
			"title": f"{full_name} - {description}",
			"url": link,
			"image_url": owner.get("avatar_url"),
		})
	return results


def fetch_youtube_links(query: str, limit: int = 3) -> List[Dict]:
	q = urllib.parse.quote_plus(query)
	return [
		{"title": f"YouTube results: {query}", "url": f"https://www.youtube.com/results?search_query={q}", "image_url": None},
		{"title": f"YouTube tutorial: {query}", "url": f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query + ' tutorial')}", "image_url": None},
		{"title": f"YouTube roadmap: {query}", "url": f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query + ' roadmap')}", "image_url": None},
	][:limit]


def fetch_website_links(query: str, limit: int = 3) -> List[Dict]:
	q = urllib.parse.quote_plus(query)
	return [
		{"title": f"DuckDuckGo: {query}", "url": f"https://duckduckgo.com/?q={q}", "image_url": None},
		{"title": f"Wikipedia search: {query}", "url": f"https://en.wikipedia.org/w/index.php?search={q}", "image_url": None},
		{"title": f"StackOverflow search: {query}", "url": f"https://stackoverflow.com/search?q={q}", "image_url": None},
	][:limit]


def fetch_transition_resources(from_skill: str, to_skill: str) -> Dict[str, List[ResourceItem]]:
	edge_query = f"{from_skill} to {to_skill}"
	practical_query = f"learn {to_skill} with {from_skill}"
	research_query = f"{to_skill} {from_skill}"

	papers_raw = fetch_arxiv(research_query, limit=3)
	books_raw = fetch_google_books(f"{to_skill} best book", limit=3)
	github_raw = fetch_github(f"{to_skill} {from_skill}", limit=3)
	wiki_raw = fetch_wikipedia(f"{to_skill} {from_skill}", limit=3)
	stack_raw = fetch_stackexchange(f"{to_skill} {from_skill}", limit=3)
	yt_raw = fetch_youtube_links(practical_query, limit=3)
	websites_raw = fetch_website_links(edge_query, limit=3)

	resources = {
		"research_papers": _to_resource_items(papers_raw, research_query, "arXiv"),
		"books": _to_resource_items(books_raw, f"{to_skill} best book", "Google Books"),
		"github": _to_resource_items(github_raw, f"{to_skill} {from_skill}", "GitHub"),
		"youtube": _to_resource_items(yt_raw, practical_query, "YouTube"),
		"websites": _to_resource_items(websites_raw, edge_query, "Web Search"),
		"documentation": _to_resource_items(wiki_raw + stack_raw, f"{to_skill} documentation", "Wikipedia/StackExchange"),
	}

	# Ensure each category always has at least one fallback item.
	for category, items in resources.items():
		if items:
			continue
		fallback_url = f"https://duckduckgo.com/?q={urllib.parse.quote_plus(edge_query + ' ' + category)}"
		resources[category] = [
			ResourceItem(
				title=f"Search {category} for {edge_query}",
				url=fallback_url,
				source="Fallback",
				level="intermediate",
				relevance_score=0.55,
				image_url=None,
			)
		]

	return resources


app = FastAPI(
	title="Skill Transition Resource Fetcher",
	description="Fetch free learning resources for a transition from one skill to another.",
	version="1.0.0",
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
	return {"status": "ok", "service": "skill-transition-resource-fetcher"}


@app.post("/get-resources", response_model=TransitionResponse)
async def get_resources(request: TransitionRequest) -> TransitionResponse:
	from_skill = request.from_skill.strip()
	to_skill = request.to_skill.strip()
	if not from_skill or not to_skill:
		raise HTTPException(status_code=400, detail="Both 'from' and 'to' must be non-empty strings.")

	resources = fetch_transition_resources(from_skill, to_skill)
	return TransitionResponse(
		result=TransitionResources(
			edge=f"{from_skill}->{to_skill}",
			resources=resources,
		)
	)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
