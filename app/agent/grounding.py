"""Post-hoc citation grounding: every document/paper/repo/report reference in an answer must come from a tool."""

from __future__ import annotations

import re
from typing import Any

DOC_CITE_RE = re.compile(r"\[([^\[\]\n]+?\.pdf),\s*p(?:age|\.)?\s*(\d+)\]", re.IGNORECASE)
ARXIV_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arXiv:\s?)(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)
GITHUB_RE = re.compile(r"github\.com/([A-Za-z0-9-]+/[A-Za-z0-9._-]+?)(?:\.git)?(?=[/)\]\s#?]|$)")
REPORT_RE = re.compile(r"/api/reports/([a-f0-9]{32})")


def _base(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id).lower()


def check_citations(answer: str, evidence: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    docs: set[tuple[str, int]] = set()
    papers: set[str] = set()
    repos: set[str] = set()
    reports: set[str] = set()
    for result in evidence.get("query_documents", []):
        for p in result.get("passages", []):
            docs.add((str(p.get("filename", "")).lower(), int(p.get("page", -1))))
    for result in evidence.get("search_papers", []):
        papers |= {_base(p.get("arxiv_id", "")) for p in result.get("papers", [])}
        if result.get("related_to"):
            papers.add(_base(result["related_to"]))
    for result in evidence.get("analyze_repository", []):
        repos.add(str(result.get("repository", "")).lower())
    for result in evidence.get("generate_report", []):
        reports.add(result.get("report_id", ""))
        for ref in result.get("references", []):
            if ref.get("source_type") == "document" and ref.get("filename") and ref.get("page") is not None:
                docs.add((str(ref["filename"]).lower(), int(ref["page"])))
            url = ref.get("url") or ""
            if m := ARXIV_RE.search(url):
                papers.add(_base(m.group(1)))
            if m := GITHUB_RE.search(url):
                repos.add(m.group(1).lower())

    verified: list[str] = []
    unverified: list[str] = []

    def record(label: str, ok: bool) -> None:
        target = verified if ok else unverified
        if label not in target:
            target.append(label)

    for m in DOC_CITE_RE.finditer(answer):
        record(f"[{m.group(1)}, p. {m.group(2)}]", (m.group(1).strip().lower(), int(m.group(2))) in docs)
    for m in ARXIV_RE.finditer(answer):
        record(f"arXiv:{m.group(1)}", _base(m.group(1)) in papers)
    for m in GITHUB_RE.finditer(answer):
        repo = m.group(1).rstrip(".")
        record(f"github.com/{repo}", repo.lower() in repos)
    for m in REPORT_RE.finditer(answer):
        record(f"report {m.group(1)}", m.group(1) in reports)
    return {"verified": verified, "unverified": unverified}
