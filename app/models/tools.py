"""Structured result schemas returned by the MCP tools (exposed as MCP output schemas)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Paper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str = Field(description="Publication date (YYYY-MM-DD)")
    updated: str | None = None
    url: str = Field(description="Canonical arXiv abstract URL")
    pdf_url: str | None = None
    categories: list[str] = []


class PaperSearchResult(BaseModel):
    query: str
    total_results: int
    papers: list[Paper]
    related_to: str | None = None
    message: str | None = None
    source: str = "arXiv API (export.arxiv.org)"


class CodeReference(BaseModel):
    path: str
    line: int | None = None
    symbol: str | None = None
    kind: str
    url: str


class RepositoryFile(BaseModel):
    path: str
    size: int
    url: str
    excerpt: str


class RepositoryAnalysis(BaseModel):
    repository: str
    url: str
    description: str | None
    default_branch: str
    commit_sha: str | None
    stars: int
    forks: int
    open_issues: int
    license: str | None
    topics: list[str]
    created_at: str | None
    pushed_at: str | None
    languages: dict[str, float] = Field(description="Language share in percent of bytes")
    readme_excerpt: str | None
    file_count: int
    tree_truncated: bool
    top_level: list[str]
    dependency_files: list[str]
    dependencies: list[str]
    ml_frameworks: list[str]
    key_algorithms: list[str]
    architecture_summary: list[str]
    selected_files: list[RepositoryFile]
    code_references: list[CodeReference]
    rate_limit_remaining: int | None = None
    notes: list[str] = []


class Passage(BaseModel):
    citation: str = Field(description="Citation key: [filename, p. N]")
    document_id: str
    filename: str
    title: str
    page: int
    chunk_index: int
    score: float = Field(description="Cosine similarity (1 - distance)")
    text: str


class DocumentQueryResult(BaseModel):
    question: str
    evidence_available: bool
    passages: list[Passage]
    documents_indexed: int
    message: str


class GroupStats(BaseModel):
    group: str
    runs: int
    final_mean: float | None
    final_std: float | None
    best_observed: float | None
    best_run: str | None
    best_epoch: float | None
    improvement_vs_baseline_pct: float | None


class MetricComparison(BaseModel):
    metric: str
    direction: str = Field(description="'lower' or 'higher' is better")
    baseline: str | None
    groups: list[GroupStats]
    winner: str | None


class ExperimentAnalysis(BaseModel):
    files: list[str]
    rows: int
    columns: list[str]
    group_by: str | None
    run_key: list[str]
    epoch_column: str | None
    filters_applied: dict[str, str]
    metric_columns: list[str]
    comparisons: list[MetricComparison]
    warnings: list[str] = []
    note: str = "All values are computed directly from the uploaded files; nothing is estimated."


class ReportReference(BaseModel):
    key: str
    label: str
    url: str | None = None
    source_type: str = Field(description="paper, document, repository or experiment_file")
    filename: str | None = None
    page: int | None = None


class ReportResult(BaseModel):
    report_id: str
    title: str
    markdown_url: str
    html_url: str
    sections: list[str]
    references: list[ReportReference]
    warnings: list[str] = []
