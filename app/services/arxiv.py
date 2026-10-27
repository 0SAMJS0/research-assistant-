"""Client for the public arXiv Atom API with rate limiting, retries and caching."""

from __future__ import annotations

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from datetime import date

import httpx

from app.core.config import Settings
from app.models.tools import Paper, PaperSearchResult
from app.services.http import ExternalServiceError, build_client

ATOM = "{http://www.w3.org/2005/Atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
ARXIV_HOSTS = {"export.arxiv.org", "arxiv.org"}
ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?/\d{7})(v\d+)?$", re.IGNORECASE)
FIELD_PREFIX_RE = re.compile(r"\b(ti|au|abs|co|jr|cat|rn|id|all):", re.IGNORECASE)
STOPWORDS = {
    "with",
    "from",
    "that",
    "this",
    "using",
    "towards",
    "toward",
    "into",
    "over",
    "under",
    "their",
    "based",
    "via",
    "for",
    "and",
    "the",
    "are",
    "learning",
    "approach",
    "method",
    "methods",
    "study",
}
MAX_RESULTS_CAP = 50
_CACHE_TTL_S = 600.0


def normalize_arxiv_id(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^https?://(export\.)?arxiv\.org/(abs|pdf)/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^arxiv:", "", value, flags=re.IGNORECASE)
    if not ARXIV_ID_RE.fullmatch(value):
        raise ValueError(f"Not a valid arXiv identifier: {value!r}")
    return value


def build_search_query(query: str, start_date: date | None = None, end_date: date | None = None) -> str:
    query = " ".join(query.replace('"', " ").split())
    if not query:
        raise ValueError("query must not be empty")
    if FIELD_PREFIX_RE.search(query):
        search = query
    else:
        terms = [t for t in re.split(r"\s+", query) if t]
        search = " AND ".join(f"all:{t}" for t in terms)
    if start_date or end_date:
        start = (start_date or date(1991, 1, 1)).strftime("%Y%m%d") + "0000"
        end = (end_date or date.today()).strftime("%Y%m%d") + "2359"
        search = f"({search}) AND submittedDate:[{start} TO {end}]"
    return search


def _text(node: ET.Element | None) -> str:
    return " ".join((node.text or "").split()) if node is not None else ""


def parse_feed(xml_text: str) -> tuple[int, list[Paper]]:
    try:
        root = ET.fromstring(xml_text)  # noqa: S314
    except ET.ParseError as exc:
        raise ExternalServiceError("arXiv returned malformed XML") from exc
    total = int(_text(root.find(f"{OPENSEARCH}totalResults")) or 0)
    papers: list[Paper] = []
    for entry in root.findall(f"{ATOM}entry"):
        entry_id = _text(entry.find(f"{ATOM}id"))
        title = _text(entry.find(f"{ATOM}title"))
        if not entry_id or "api/errors" in entry_id or title.lower() == "error":
            raise ExternalServiceError(f"arXiv rejected the query: {_text(entry.find(f'{ATOM}summary'))}")
        raw_id = entry_id.rsplit("/abs/", 1)[-1]
        base_id = re.sub(r"v\d+$", "", raw_id)
        pdf_url = None
        for link in entry.findall(f"{ATOM}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
        papers.append(
            Paper(
                arxiv_id=raw_id,
                title=title,
                authors=[_text(a.find(f"{ATOM}name")) for a in entry.findall(f"{ATOM}author")],
                abstract=_text(entry.find(f"{ATOM}summary")),
                published=_text(entry.find(f"{ATOM}published"))[:10],
                updated=_text(entry.find(f"{ATOM}updated"))[:10] or None,
                url=f"https://arxiv.org/abs/{base_id}",
                pdf_url=pdf_url,
                categories=[c.get("term", "") for c in entry.findall(f"{ATOM}category") if c.get("term")],
            )
        )
    return total, papers


class ArxivClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport
        self._lock = asyncio.Lock()
        self._last_request = 0.0
        self._cache: dict[str, tuple[float, tuple[int, list[Paper]]]] = {}

    async def _get(self, params: dict[str, str | int]) -> tuple[int, list[Paper]]:
        cache_key = repr(sorted(params.items()))
        cached = self._cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < _CACHE_TTL_S:
            return cached[1]

        base, _, path = self._settings.arxiv_api_url.partition("/api")
        client = build_client(base, ARXIV_HOSTS, self._settings.http_timeout_s, transport=self._transport)
        path = "/api" + path
        last_error = "unknown error"
        async with client:
            for attempt in range(3):
                async with self._lock:
                    # arXiv asks clients to wait ~3 s between requests.
                    wait = self._settings.arxiv_min_interval_s - (time.monotonic() - self._last_request)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self._last_request = time.monotonic()
                    try:
                        response = await client.get(path, params=params)
                    except httpx.TimeoutException:
                        last_error = "request to arXiv timed out"
                        continue
                    except httpx.HTTPError as exc:
                        last_error = f"network error contacting arXiv ({type(exc).__name__})"
                        continue
                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"arXiv returned HTTP {response.status_code}"
                    retry_after = response.headers.get("Retry-After", "")
                    delay = float(retry_after) if retry_after.isdigit() else 2.0 * (attempt + 1)
                    if self._settings.arxiv_min_interval_s > 0:  # zero interval = tests, no backoff
                        await asyncio.sleep(min(delay, 10.0))
                    continue
                if response.status_code >= 400:
                    raise ExternalServiceError(f"arXiv returned HTTP {response.status_code}")
                result = parse_feed(response.text)
                self._cache[cache_key] = (time.monotonic(), result)
                return result
        raise ExternalServiceError(f"arXiv request failed after retries: {last_error}")

    async def search(
        self,
        query: str,
        max_results: int = 10,
        start_date: date | None = None,
        end_date: date | None = None,
        sort_by: str = "relevance",
    ) -> PaperSearchResult:
        max_results = max(1, min(max_results, MAX_RESULTS_CAP))
        search_query = build_search_query(query, start_date, end_date)
        total, papers = await self._get(
            {
                "search_query": search_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": "descending",
            }
        )
        message = None if papers else "No arXiv papers matched this query and date range."
        return PaperSearchResult(query=search_query, total_results=total, papers=papers, message=message)

    async def get_papers(self, arxiv_ids: list[str]) -> list[Paper]:
        ids = [normalize_arxiv_id(i) for i in arxiv_ids][:MAX_RESULTS_CAP]
        if not ids:
            return []
        _, papers = await self._get({"id_list": ",".join(ids), "max_results": len(ids)})
        return papers

    async def related(
        self,
        arxiv_id: str,
        max_results: int = 10,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> PaperSearchResult:
        seed_papers = await self.get_papers([arxiv_id])
        if not seed_papers:
            return PaperSearchResult(
                query=arxiv_id,
                total_results=0,
                papers=[],
                related_to=arxiv_id,
                message=f"arXiv has no paper with id {arxiv_id}.",
            )
        seed = seed_papers[0]
        words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", seed.title)]
        keywords = [w for w in dict.fromkeys(words) if w not in STOPWORDS][:5]
        if not keywords:
            keywords = words[:3]
        query = " OR ".join(f"ti:{k}" for k in keywords) or f"all:{seed.title}"
        result = await self.search(f"({query})", max_results + 1, start_date, end_date)
        seed_base = re.sub(r"v\d+$", "", seed.arxiv_id)
        result.papers = [p for p in result.papers if re.sub(r"v\d+$", "", p.arxiv_id) != seed_base][
            :max_results
        ]
        result.related_to = seed.arxiv_id
        if not result.papers:
            result.message = f"No related papers found for {seed.arxiv_id}."
        return result
