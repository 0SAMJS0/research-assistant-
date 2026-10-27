"""Application service container stored on `app.state`."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.agent.graph import ResearchAgent
from app.agent.mcp_client import MCPToolProvider
from app.core.config import Settings
from app.rag.store import DocumentStore
from app.services.reports import ReportWriter


@dataclass
class Container:
    settings: Settings
    store: DocumentStore
    reports: ReportWriter
    tool_provider: MCPToolProvider
    agent: ResearchAgent


def get_container(request: Request) -> Container:
    return request.app.state.container
