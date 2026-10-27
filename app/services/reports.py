"""Cited research report generation (Markdown + sanitized HTML) from verified evidence only."""

from __future__ import annotations

import html
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from markdown_it import MarkdownIt

from app.core.security import REPORT_ID_RE, safe_child
from app.models.tools import (
    DocumentQueryResult,
    ExperimentAnalysis,
    Paper,
    ReportReference,
    ReportResult,
    RepositoryAnalysis,
)

_MD_SPECIAL = re.compile(r"([\\`*\[\]<>#|~])")  # enough to stop links, HTML, emphasis and tables
_URL_RE = re.compile(r"https?://[^\s)\]>]+")
_ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(v\d+)?\b")
_md = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")


def md_escape(text: str | None, limit: int | None = None) -> str:
    """Neutralize Markdown/HTML in untrusted text (paper abstracts, README, PDF passages)."""
    value = " ".join((text or "").split())
    if limit and len(value) > limit:
        value = value[:limit].rsplit(" ", 1)[0] + " …"
    return _MD_SPECIAL.sub(r"\\\1", value)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def _first_sentences(text: str, count: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return " ".join(parts[:count])


@dataclass
class ReportInputs:
    title: str
    research_question: str
    papers: list[Paper] = field(default_factory=list)
    documents: DocumentQueryResult | None = None
    repository: RepositoryAnalysis | None = None
    experiments: ExperimentAnalysis | None = None
    interpretation: str | None = None
    warnings: list[str] = field(default_factory=list)


class _Refs:
    def __init__(self) -> None:
        self.items: list[ReportReference] = []
        self._index: dict[str, int] = {}

    def cite(self, key: str, label: str, url: str | None = None, **source: str | int | None) -> str:
        if key not in self._index:
            source.setdefault("source_type", key.split(":", 1)[0])
            self.items.append(ReportReference(key=str(len(self.items) + 1), label=label, url=url, **source))
            self._index[key] = len(self.items)
        return f"[{self._index[key]}]"


def sanitize_interpretation(
    text: str, allowed_urls: set[str], allowed_ids: set[str]
) -> tuple[str, list[str]]:
    """Remove URLs / arXiv IDs from model text that are not among the report's verified references."""
    warnings: list[str] = []

    def _url(match: re.Match[str]) -> str:
        url = match.group(0).rstrip(".,;")
        if url in allowed_urls:
            return match.group(0)
        warnings.append(f"Removed unverified URL from interpretation: {url[:80]}")
        return "[unverified link removed]"

    def _id(match: re.Match[str]) -> str:
        base = re.sub(r"v\d+$", "", match.group(0))
        if base in allowed_ids:
            return match.group(0)
        warnings.append(f"Removed unverified arXiv identifier from interpretation: {match.group(0)}")
        return "[unverified reference removed]"

    cleaned = _URL_RE.sub(_url, text)
    cleaned = _ARXIV_ID_RE.sub(_id, cleaned)
    return cleaned, warnings


def build_markdown(
    inputs: ReportInputs, report_id: str
) -> tuple[str, list[ReportReference], list[str], list[str]]:
    refs = _Refs()
    warnings = list(inputs.warnings)
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = []
    out: list[str] = [
        f"# {md_escape(inputs.title, 200)}",
        "",
        f"*Generated {now} by MCP Research Agent · Report ID `{report_id}`*",
        "",
        "> **How to read this report.** Statements tagged **[Observed]** are copied from the cited source or "
        "computed directly from uploaded data. The *Model-generated interpretation* section is written by a "
        "language model and is not evidence. Every factual claim carries a numbered citation to the References.",
        "",
    ]

    paper_cites = {
        p.arxiv_id: refs.cite(
            f"paper:{p.arxiv_id}",
            f"{', '.join(p.authors[:3])}{' et al.' if len(p.authors) > 3 else ''}. "
            f'"{p.title}". arXiv:{p.arxiv_id} ({p.published[:4]}).',
            p.url,
        )
        for p in inputs.papers
    }
    doc_cites: dict[tuple[str, int], str] = {}
    if inputs.documents and inputs.documents.evidence_available:
        for passage in inputs.documents.passages:
            doc_cites[(passage.document_id, passage.page)] = refs.cite(
                f"document:{passage.document_id}:{passage.page}",
                f"{passage.title} (uploaded file {passage.filename}), page {passage.page}.",
                filename=passage.filename,
                page=passage.page,
            )
    repo_cite = (
        refs.cite(
            f"repository:{inputs.repository.repository}",
            f"GitHub repository {inputs.repository.repository}"
            + (f" at commit {inputs.repository.commit_sha[:12]}" if inputs.repository.commit_sha else "")
            + ".",
            inputs.repository.url,
        )
        if inputs.repository
        else None
    )
    exp_cites = {
        name: refs.cite(
            f"experiment_file:{name}", f"Uploaded experiment log {name} (user-provided data).", filename=name
        )
        for name in (inputs.experiments.files if inputs.experiments else [])
    }
    exp_cite_all = " ".join(exp_cites.values())

    # Executive summary
    sections.append("Executive Summary")
    out += ["## Executive Summary", ""]
    if inputs.papers:
        out.append(
            f"- **[Observed]** {len(inputs.papers)} arXiv paper(s) were retrieved as related work "
            f"{' '.join(paper_cites.values())}."
        )
    if inputs.documents is not None:
        if inputs.documents.evidence_available:
            n_docs = len({p.document_id for p in inputs.documents.passages})
            out.append(
                f"- **[Observed]** {len(inputs.documents.passages)} relevant passage(s) from {n_docs} uploaded "
                f"document(s) were found {' '.join(dict.fromkeys(doc_cites.values()))}."
            )
        else:
            out.append(
                "- **[Observed]** Evidence unavailable: no uploaded document passage supports this "
                "research question."
            )
    if inputs.repository:
        fw = ", ".join(inputs.repository.ml_frameworks) or "no recognized ML framework"
        out.append(
            f"- **[Observed]** Repository `{md_escape(inputs.repository.repository)}` uses {md_escape(fw)} "
            f"{repo_cite}."
        )
    if inputs.experiments:
        for comp in inputs.experiments.comparisons[:4]:
            if comp.winner:
                win = next(g for g in comp.groups if g.group == comp.winner)
                base = next((g for g in comp.groups if g.group == comp.baseline), None)
                line = (
                    f"- **[Observed]** On `{md_escape(comp.metric)}` ({comp.direction} is better), "
                    f"`{md_escape(win.group)}` had the best mean final value {_fmt(win.final_mean)} "
                    f"over {win.runs} run(s)"
                )
                if base and base.group != win.group:
                    line += (
                        f" vs. baseline `{md_escape(base.group)}` at {_fmt(base.final_mean)} "
                        f"({_fmt(win.improvement_vs_baseline_pct, 3)}% improvement)"
                    )
                out.append(line + f" {exp_cite_all}.")
    if len(out) and not out[-1].startswith("- "):
        out.append("- No evidence sources were supplied.")
    out.append("")

    sections.append("Research Question")
    out += ["## Research Question", "", md_escape(inputs.research_question, 2000), ""]

    sections.append("Related Work")
    out += ["## Related Work", ""]
    if inputs.papers:
        for p in inputs.papers:
            authors = ", ".join(p.authors[:4]) + (" et al." if len(p.authors) > 4 else "")
            out.append(
                f"- **{md_escape(p.title)}** — {md_escape(authors)} ({p.published}). "
                f"Abstract (excerpt): {md_escape(_first_sentences(p.abstract), 600)} {paper_cites[p.arxiv_id]}"
            )
    else:
        out.append("No papers were retrieved for this report.")
    out.append("")

    sections.append("Methodology")
    out += ["## Methodology", ""]
    method = []
    if inputs.papers:
        method.append(
            "Related work was retrieved live from the arXiv API; titles, authors, dates and abstracts are "
            "reproduced from arXiv metadata."
        )
    if inputs.documents is not None:
        method.append(
            "Uploaded PDFs were split into page-aware chunks, embedded, and searched semantically in "
            "ChromaDB; only passages above the relevance threshold are cited, with page numbers."
        )
    if inputs.repository:
        method.append(
            "The repository was inspected read-only through the GitHub REST API (metadata, languages, "
            "file tree, README, dependency manifests and selected source files); no code was executed."
        )
    if inputs.experiments:
        method.append(
            "Experiment logs were parsed with pandas. For each run the final value is the metric at the "
            "last logged epoch; groups report mean and sample standard deviation of final values across "
            "runs, the best value observed at any epoch, and direction-aware percentage improvement of "
            "the mean final value relative to the baseline group."
        )
    out += [f"- {m}" for m in method] or ["- No evidence sources were consulted."]
    out.append("")

    if inputs.documents is not None:
        sections.append("Document Evidence")
        out += ["## Document Evidence", ""]
        if inputs.documents.evidence_available:
            for p in inputs.documents.passages:
                out.append(
                    f"- **[Observed]** “{md_escape(p.text, 450)}” — *{md_escape(p.filename)}*, "
                    f"p. {p.page} (similarity {p.score:.2f}) {doc_cites[(p.document_id, p.page)]}"
                )
        else:
            out.append(
                "Evidence unavailable: the uploaded documents contain no passage relevant to the question."
            )
        out.append("")

    if inputs.repository:
        r = inputs.repository
        sections.append("Repository Analysis")
        out += ["## Repository Analysis", ""]
        out.append(
            f"- **[Observed]** {md_escape(r.description or 'No description.', 300)} "
            f"Stars: {r.stars}; license: {md_escape(r.license or 'none declared')} {repo_cite}"
        )
        out += [f"- **[Observed]** {md_escape(line, 400)} {repo_cite}" for line in r.architecture_summary]
        if r.key_algorithms:
            out.append(
                f"- **[Observed]** Algorithm keywords found in source: {md_escape(', '.join(r.key_algorithms))} "
                f"{repo_cite}"
            )
        for ref in r.code_references[:8]:
            out.append(
                f"  - `{md_escape(ref.symbol or '')}` ({ref.kind}) — [{md_escape(ref.path)}:{ref.line}]({ref.url})"
            )
        out.append("")

    sections.append("Experimental Results")
    out += ["## Experimental Results", ""]
    if inputs.experiments and inputs.experiments.comparisons:
        e = inputs.experiments
        out.append(
            f"**[Observed]** Computed from {e.rows} rows in {', '.join(f'`{md_escape(f)}`' for f in e.files)} "
            f"{exp_cite_all}. Grouped by `{md_escape(e.group_by or 'all rows')}`"
            + (f"; filters: {md_escape(str(e.filters_applied))}" if e.filters_applied else "")
            + "."
        )
        out.append("")
        for comp in e.comparisons:
            out += [
                f"### {md_escape(comp.metric)} ({comp.direction} is better; baseline: "
                f"{md_escape(comp.baseline or 'n/a')})",
                "",
                "| Group | Runs | Final mean | Final std | Best observed | Best epoch | Δ vs baseline (%) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for g in comp.groups:
                out.append(
                    f"| {md_escape(g.group)} | {g.runs} | {_fmt(g.final_mean)} | {_fmt(g.final_std)} | "
                    f"{_fmt(g.best_observed)} | {_fmt(g.best_epoch)} | {_fmt(g.improvement_vs_baseline_pct, 3)} |"
                )
            out.append("")
        if e.warnings:
            out += [f"- Data note: {md_escape(w, 300)}" for w in e.warnings] + [""]
    else:
        out += ["No experiment results were supplied.", ""]

    sections.append("Comparison Table")
    out += ["## Comparison Table", ""]
    if inputs.experiments and inputs.experiments.comparisons:
        out += [
            "| Metric | Better | Baseline | Baseline mean | Best group | Best mean | Δ (%) |",
            "|---|---|---|---:|---|---:|---:|",
        ]
        for comp in inputs.experiments.comparisons:
            base = next((g for g in comp.groups if g.group == comp.baseline), None)
            win = next((g for g in comp.groups if g.group == comp.winner), None)
            out.append(
                f"| {md_escape(comp.metric)} | {comp.direction} | {md_escape(comp.baseline or 'n/a')} | "
                f"{_fmt(base.final_mean if base else None)} | {md_escape(comp.winner or 'n/a')} | "
                f"{_fmt(win.final_mean if win else None)} | "
                f"{_fmt(win.improvement_vs_baseline_pct if win else None, 3)} |"
            )
        out.append(f"\nSource: {exp_cite_all}")
    elif inputs.papers:
        out += ["| # | Paper | First author | Published | Primary category |", "|---:|---|---|---|---|"]
        for p in inputs.papers:
            out.append(
                f"| {paper_cites[p.arxiv_id]} | {md_escape(p.title, 120)} | "
                f"{md_escape(p.authors[0] if p.authors else 'n/a')} | {p.published} | "
                f"{md_escape(p.categories[0] if p.categories else 'n/a')} |"
            )
    else:
        out.append("No comparable quantities were available.")
    out.append("")

    if inputs.interpretation and inputs.interpretation.strip():
        allowed_urls = {r.url for r in refs.items if r.url}
        allowed_ids = {re.sub(r"v\d+$", "", p.arxiv_id) for p in inputs.papers}
        cleaned, removed = sanitize_interpretation(inputs.interpretation, allowed_urls, allowed_ids)
        warnings += removed
        sections.append("Model-generated Interpretation")
        out += [
            "## Model-generated Interpretation",
            "",
            "> ⚠️ Written by the language model from the evidence above. It is interpretation, not "
            "measurement; verify against the cited sources.",
            "",
            md_escape(cleaned, 6000),
            "",
        ]

    sections.append("Limitations")
    out += ["## Limitations", ""]
    limits = [
        "Coverage is limited to the sources listed in References; absence of evidence here is not evidence "
        "of absence."
    ]
    if inputs.papers:
        limits.append(
            "Related work reflects arXiv search ranking for the given query and is not a systematic review."
        )
    if inputs.documents is not None:
        limits.append(
            "Document retrieval is semantic similarity search over extracted text; figures, tables "
            "rendered as images, and scanned pages are not covered."
        )
    if inputs.repository:
        limits.append(
            "Repository analysis inspects a bounded subset of files with static pattern matching and "
            "does not execute or test the code."
        )
    if inputs.experiments:
        limits.append(
            "Experiment statistics describe only the uploaded logs; small numbers of seeds limit "
            "statistical confidence, and no significance testing is performed."
        )
        runs = [g.runs for c in inputs.experiments.comparisons for g in c.groups]
        if runs and min(runs) < 3:
            limits.append(
                "At least one group has fewer than 3 runs, so standard deviations are unreliable or "
                "unavailable."
            )
    out += [f"- {line}" for line in limits] + [""]

    sections.append("References")
    out += ["## References", ""]
    if refs.items:
        for ref in refs.items:
            label = md_escape(ref.label, 400)
            out.append(f"{ref.key}. {label}" + (f" <{ref.url}>" if ref.url else ""))
    else:
        out.append("No references.")
    out.append("")
    return "\n".join(out), refs.items, sections, warnings


REPORT_CSS = """
body{font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;max-width:860px;margin:2rem auto;
  padding:0 1rem;color:#1f2328;background:#fff}
h1,h2,h3{line-height:1.25}
h2{border-bottom:1px solid #d0d7de;padding-bottom:.3rem;margin-top:2rem}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:1rem 0}
th,td{border:1px solid #d0d7de;padding:.4rem .6rem}
th{background:#f6f8fa}
blockquote{margin:0;padding:.5rem 1rem;border-left:4px solid #d4a72c;background:#fff8c5}
code{background:#f6f8fa;padding:.1rem .3rem;border-radius:4px}
a{color:#0969da}
@media (prefers-color-scheme: dark){
  body{background:#0d1117;color:#e6edf3}
  th{background:#161b22}
  th,td{border-color:#30363d}
  code{background:#161b22}
  blockquote{background:#272115;border-color:#9e6a03}
  a{color:#4493f8}
}
"""


def render_html(markdown_text: str, title: str) -> str:
    body = _md.render(markdown_text)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{REPORT_CSS}</style></head><body>
{body}
</body></html>
"""


class ReportWriter:
    def __init__(self, reports_dir: Path, public_base_url: str) -> None:
        self.reports_dir = reports_dir
        self.public_base_url = public_base_url.rstrip("/")

    def write(self, inputs: ReportInputs) -> ReportResult:
        report_id = uuid.uuid4().hex
        markdown_text, references, sections, warnings = build_markdown(inputs, report_id)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        safe_child(self.reports_dir, f"{report_id}.md").write_text(markdown_text, encoding="utf-8")
        safe_child(self.reports_dir, f"{report_id}.html").write_text(
            render_html(markdown_text, inputs.title), encoding="utf-8"
        )
        return ReportResult(
            report_id=report_id,
            title=inputs.title,
            markdown_url=f"{self.public_base_url}/api/reports/{report_id}?format=md",
            html_url=f"{self.public_base_url}/api/reports/{report_id}?format=html",
            sections=sections,
            references=references,
            warnings=warnings,
        )

    def path_for(self, report_id: str, fmt: str) -> Path | None:
        if not REPORT_ID_RE.fullmatch(report_id) or fmt not in ("md", "html"):
            return None
        path = safe_child(self.reports_dir, f"{report_id}.{fmt}")
        return path if path.is_file() else None
