from __future__ import annotations

import warnings

import httpx
import pytest

from app.core.config import Settings
from app.mcp_server.server import ToolServices
from app.rag.store import DocumentStore, make_chroma_client
from app.services.arxiv import ArxivClient
from app.services.github import GitHubClient
from app.services.reports import ReportWriter
from tests.helpers import arxiv_handler, github_handler, make_settings

warnings.filterwarnings("ignore", message=".*langchain.mcp.*")


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = make_settings(tmp_path)
    s.ensure_dirs()
    return s


@pytest.fixture
def chroma_client(settings):
    client = make_chroma_client(settings)
    client.reset()
    return client


@pytest.fixture
def store(settings, chroma_client) -> DocumentStore:
    return DocumentStore(settings, client=chroma_client)


@pytest.fixture
def services(settings, store) -> ToolServices:
    return ToolServices(
        settings=settings,
        arxiv=ArxivClient(settings, transport=httpx.MockTransport(arxiv_handler())),
        github=GitHubClient(settings, transport=httpx.MockTransport(github_handler())),
        store=store,
        reports=ReportWriter(settings.reports_dir, settings.public_base_url),
    )
