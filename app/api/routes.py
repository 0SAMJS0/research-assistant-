"""REST API routes."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from app.agent.graph import AgentUnavailableError
from app.api.container import Container, get_container
from app.core.security import require_api_key, safe_child, sanitize_filename
from app.models.api import (
    ChatRequest,
    ChatResponse,
    Citation,
    CitationCheck,
    DocumentInfo,
    HealthResponse,
    PublicConfig,
    ReportLink,
    ToolInfo,
    ToolsResponse,
    ToolTraceEntry,
    UploadExperimentResponse,
    UploadPDFResponse,
)
from app.models.tools import ExperimentAnalysis
from app.rag.pdf import PDFError
from app.services import experiments as exp

logger = logging.getLogger(__name__)

health_router = APIRouter(tags=["health"])
public_router = APIRouter(prefix="/api", tags=["meta"])
router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

PDF_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream"}
EXPERIMENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/json",
    "text/plain",
    "application/vnd.ms-excel",
    "application/octet-stream",
}


async def read_limited(upload: UploadFile, limit: int) -> bytes:
    data = bytearray()
    while chunk := await upload.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE, f"File exceeds {limit // (1024 * 1024)} MB limit"
            )
    return bytes(data)


@health_router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@health_router.get(
    "/readyz",
    response_model=HealthResponse,
    summary="Readiness probe (ChromaDB, MCP, storage)",
    responses={503: {"model": HealthResponse}},
)
async def readyz(c: Container = Depends(get_container)) -> Any:
    checks: dict[str, str] = {}
    try:
        await asyncio.wait_for(asyncio.to_thread(c.store.heartbeat), timeout=5)
        checks["chromadb"] = "ok"
    except Exception as exc:
        checks["chromadb"] = f"error: {type(exc).__name__}"
    try:
        tools = await asyncio.wait_for(c.tool_provider.describe(), timeout=10)
        checks["mcp"] = f"ok ({len(tools)} tools)"
    except Exception as exc:
        checks["mcp"] = f"error: {type(exc).__name__}"
    for name, path in (("data_dir", c.settings.data_dir), ("reports_dir", c.settings.reports_dir)):
        checks[name] = "ok" if path.is_dir() and os.access(path, os.W_OK) else "error: not writable"
    checks["llm"] = "configured" if c.agent.available else "not configured (chat disabled)"
    ok = all(v.startswith("ok") for k, v in checks.items() if k != "llm")
    body = HealthResponse(status="ready" if ok else "not ready", checks=checks)
    return body if ok else JSONResponse(body.model_dump(), status_code=503)


@public_router.get("/config", response_model=PublicConfig, summary="Public UI configuration")
async def public_config(c: Container = Depends(get_container)) -> PublicConfig:
    return PublicConfig(
        auth_required=bool(c.settings.secret("app_api_key")),
        llm_configured=c.agent.available,
        model=c.settings.openai_model if c.agent.available else None,
        max_upload_mb=c.settings.max_upload_mb,
        environment=c.settings.app_env,
    )


def _citations(results: dict[str, list[dict[str, Any]]]) -> list[Citation]:
    out: list[Citation] = []
    seen: set[tuple[str, str]] = set()

    def add(c: Citation) -> None:
        key = (c.type, c.label)
        if key not in seen:
            seen.add(key)
            out.append(c)

    for result in results.get("search_papers", []):
        for p in result.get("papers", []):
            authors = ", ".join(p.get("authors", [])[:3]) + (
                " et al." if len(p.get("authors", [])) > 3 else ""
            )
            add(
                Citation(
                    type="paper",
                    label=p["title"],
                    url=p.get("url"),
                    detail=f"{authors} · {p.get('published', '')} · arXiv:{p.get('arxiv_id')}",
                )
            )
    for result in results.get("query_documents", []):
        for p in result.get("passages", []):
            add(Citation(type="document", label=p["citation"], detail=p.get("text", "")[:300]))
    for result in results.get("analyze_repository", []):
        add(
            Citation(
                type="repository",
                label=result["repository"],
                url=result.get("url"),
                detail=result.get("description"),
            )
        )
        for ref in result.get("code_references", [])[:6]:
            add(
                Citation(
                    type="code",
                    label=f"{ref['path']}:{ref.get('line')} {ref.get('symbol') or ''}".strip(),
                    url=ref.get("url"),
                    detail=ref.get("kind"),
                )
            )
    for result in results.get("fetch_experiments", []):
        for name in result.get("files", []):
            add(Citation(type="experiment_file", label=name, detail="Uploaded experiment log"))
    return out


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask the research agent",
    responses={503: {"description": "LLM not configured"}, 504: {"description": "Timed out"}},
)
async def chat(body: ChatRequest, c: Container = Depends(get_container)) -> ChatResponse:
    try:
        result = await c.agent.run(body.message, body.thread_id)
    except AgentUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except Exception as exc:
        logger.exception("Agent run failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"The agent failed to complete the request ({type(exc).__name__})."
        ) from exc
    reports = [
        ReportLink(**{k: r[k] for k in ("report_id", "title", "markdown_url", "html_url")})
        for r in result.tool_results.get("generate_report", [])
    ]
    tables = [ExperimentAnalysis.model_validate(r) for r in result.tool_results.get("fetch_experiments", [])]
    return ChatResponse(
        thread_id=result.thread_id,
        answer=result.answer,
        tool_trace=[
            ToolTraceEntry(
                **{k: e.get(k) for k in ("tool", "status", "duration_ms", "error")},
                arguments=e.get("arguments") or {},
            )
            for e in result.trace
        ],
        citations=_citations(result.tool_results),
        experiment_tables=tables,
        reports=reports,
        citation_check=CitationCheck(**result.citation_check),
        stopped_reason=result.stopped_reason,
    )


@router.post(
    "/upload/pdf",
    response_model=UploadPDFResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF",
    responses={200: {"model": UploadPDFResponse}},
)
async def upload_pdf(file: UploadFile = File(...), c: Container = Depends(get_container)) -> Any:
    name = sanitize_filename(file.filename or "document.pdf", "document.pdf")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only .pdf files are accepted")
    if file.content_type and file.content_type not in PDF_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported content type {file.content_type}"
        )
    data = await read_limited(file, c.settings.max_upload_bytes)
    try:
        result = await asyncio.to_thread(c.store.ingest, data, name)
    except PDFError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    body = UploadPDFResponse(
        **result.__dict__,
        message=(
            "Duplicate upload: this exact file is already indexed."
            if result.duplicate
            else f"Indexed {result.chunks} chunks from {result.pages} page(s)."
        ),
    )
    return JSONResponse(body.model_dump(), status_code=200 if result.duplicate else 201)


@router.post(
    "/upload/experiment",
    response_model=UploadExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an experiment CSV/JSON log",
    responses={200: {"model": UploadExperimentResponse}},
)
async def upload_experiment(file: UploadFile = File(...), c: Container = Depends(get_container)) -> Any:
    name = sanitize_filename(file.filename or "experiment.csv", "experiment.csv")
    suffix = Path(name).suffix.lower()
    if suffix not in exp.ALLOWED_SUFFIXES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only .csv and .json files are accepted")
    if file.content_type and file.content_type not in EXPERIMENT_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported content type {file.content_type}"
        )
    data = await read_limited(file, c.settings.max_upload_bytes)
    try:
        df = await asyncio.to_thread(exp.parse_experiment_bytes, data, name)
    except exp.ExperimentDataError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    metrics = exp.detect_metric_columns(df, set())
    metrics = [m for m in metrics if m not in {"seed", "epoch", "step", "random_seed", "iteration"}]
    if not metrics:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "No numeric metric columns found in the file"
        )

    directory = c.settings.experiments_dir
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()
    target = safe_child(directory, name)
    duplicate = False
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            duplicate = True
        else:
            target = safe_child(directory, f"{Path(name).stem[:80]}-{digest[:8]}{suffix}")
            duplicate = target.exists()
    if not duplicate:
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".upload-")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, target)
    body = UploadExperimentResponse(
        file_name=target.name,
        rows=len(df),
        columns=list(df.columns),
        metric_columns=metrics,
        duplicate=duplicate,
        message="Duplicate upload: identical file already stored."
        if duplicate
        else "Experiment file stored.",
    )
    return JSONResponse(body.model_dump(), status_code=200 if duplicate else 201)


@router.get(
    "/reports/{report_id}",
    summary="Download a generated report",
    responses={200: {"content": {"text/markdown": {}, "text/html": {}}}, 404: {}},
)
async def get_report(
    report_id: str,
    format: Literal["md", "html"] = Query("md"),
    c: Container = Depends(get_container),
) -> FileResponse:
    path = c.reports.path_for(report_id, format)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    if format == "html":
        return FileResponse(
            path,
            media_type="text/html; charset=utf-8",
            headers={
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src https: data:; "
                "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
            },
        )
    return FileResponse(
        path, media_type="text/markdown; charset=utf-8", filename=f"research-report-{report_id}.md"
    )


@router.get("/tools", response_model=ToolsResponse, summary="List tools discovered from the MCP server")
async def list_tools(c: Container = Depends(get_container)) -> ToolsResponse:
    try:
        tools = await asyncio.wait_for(c.tool_provider.describe(), timeout=15)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"MCP server unreachable ({type(exc).__name__})"
        ) from exc
    return ToolsResponse(mcp_url=c.settings.mcp_base_url, tools=[ToolInfo(**t) for t in tools])


@router.get("/documents", response_model=list[DocumentInfo], summary="List indexed documents")
async def list_documents(c: Container = Depends(get_container)) -> list[DocumentInfo]:
    return [DocumentInfo(**d) for d in await asyncio.to_thread(c.store.list_documents)]


@router.get("/experiments", response_model=list[str], summary="List uploaded experiment files")
async def list_experiments(c: Container = Depends(get_container)) -> list[str]:
    return exp.list_experiment_files(c.settings.experiments_dir)
