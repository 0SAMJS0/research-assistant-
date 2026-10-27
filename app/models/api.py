"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.tools import ExperimentAnalysis


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1, max_length=4000, examples=["Find recent papers on graph neural networks"]
    )
    thread_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{8,64}$",
        description="Conversation ID; omit to start a new conversation",
    )


class ToolTraceEntry(BaseModel):
    tool: str
    status: Literal["success", "error", "timeout", "skipped"]
    duration_ms: float
    arguments: dict[str, Any] = {}
    error: str | None = None


class Citation(BaseModel):
    type: Literal["paper", "document", "repository", "code", "experiment_file"]
    label: str
    url: str | None = None
    detail: str | None = None


class ReportLink(BaseModel):
    report_id: str
    title: str
    markdown_url: str
    html_url: str


class CitationCheck(BaseModel):
    verified: list[str] = []
    unverified: list[str] = []


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    tool_trace: list[ToolTraceEntry]
    citations: list[Citation]
    experiment_tables: list[ExperimentAnalysis]
    reports: list[ReportLink]
    citation_check: CitationCheck
    stopped_reason: str | None = None


class UploadPDFResponse(BaseModel):
    document_id: str
    filename: str
    title: str
    sha256: str
    pages: int
    chunks: int
    duplicate: bool
    message: str


class UploadExperimentResponse(BaseModel):
    file_name: str
    rows: int
    columns: list[str]
    metric_columns: list[str]
    duplicate: bool
    message: str


class ToolInfo(BaseModel):
    name: str
    title: str | None = None
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None


class ToolsResponse(BaseModel):
    mcp_url: str
    tools: list[ToolInfo]


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    title: str
    pages: int | None = None
    chunks: int
    uploaded_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, str] = {}


class PublicConfig(BaseModel):
    auth_required: bool
    llm_configured: bool
    model: str | None
    max_upload_mb: int
    environment: str
