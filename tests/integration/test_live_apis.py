"""Optional tests against LIVE external services. Skipped unless RUN_LIVE_TESTS=1.

Requirements:
- arXiv tests: outbound HTTPS to export.arxiv.org (no credentials).
- GitHub test: outbound HTTPS to api.github.com; GITHUB_TOKEN optional (raises the quota from 60 req/h).
- Agent test: OPENAI_API_KEY (paid) and OPENAI_MODEL; uses the real LLM with the local MCP server.

    RUN_LIVE_TESTS=1 pytest tests/integration/test_live_apis.py -v
"""

import os
import shutil

import pytest

from app.agent.graph import ResearchAgent, build_chat_model
from app.agent.mcp_client import MCPToolProvider
from app.mcp_server.server import ToolServices, build_mcp_app, create_mcp_server
from app.services.arxiv import ArxivClient
from app.services.github import GitHubClient
from tests.helpers import MCP_TOKEN, SAMPLES, ServerThread, free_port, make_settings

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call live APIs"
    ),
]


async def test_live_arxiv_search_and_related(tmp_path):
    arxiv = ArxivClient(make_settings(tmp_path, arxiv_min_interval_s=3))
    result = await arxiv.search("graph neural networks", max_results=3)
    assert result.total_results > 100 and len(result.papers) == 3
    assert all(p.url.startswith("https://arxiv.org/abs/") and p.authors for p in result.papers)
    related = await arxiv.related("1609.02907", max_results=3)
    assert related.related_to.startswith("1609.02907") and related.papers


async def test_live_github_analysis(tmp_path):
    gh = GitHubClient(make_settings(tmp_path, github_token=os.environ.get("GITHUB_TOKEN") or None))
    result = await gh.analyze("https://github.com/pyg-team/pytorch_geometric")
    assert "PyTorch" in result.ml_frameworks and result.code_references


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="requires OPENAI_API_KEY")
async def test_live_agent_end_to_end(tmp_path):
    settings = make_settings(
        tmp_path,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        arxiv_min_interval_s=3,
    )
    settings.ensure_dirs()
    shutil.copy(SAMPLES / "synthetic_gnn_experiments.csv", settings.experiments_dir)
    port = free_port()
    settings.mcp_base_url = f"http://127.0.0.1:{port}/mcp"
    with ServerThread(build_mcp_app(create_mcp_server(ToolServices.from_settings(settings)), settings), port):
        agent = ResearchAgent(
            build_chat_model(settings), MCPToolProvider(settings.mcp_base_url, MCP_TOKEN), settings
        )
        result = await agent.run("Compare GCN and GAT in my uploaded experiments on val_accuracy.")
    assert any(t["tool"] == "fetch_experiments" and t["status"] == "success" for t in result.trace)
    assert "GAT" in result.answer
