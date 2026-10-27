"""MCP server over the real Streamable HTTP transport (uvicorn on localhost), with mocked arXiv/GitHub."""

import json
import shutil

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain_core.messages import AIMessage

from app.agent.graph import ResearchAgent
from app.agent.mcp_client import MCPToolProvider
from app.mcp_server.server import TOOL_NAMES, build_mcp_app, create_mcp_server
from tests.helpers import (
    MCP_TOKEN,
    SAMPLES,
    ScriptedChatModel,
    ServerThread,
    free_port,
    last_tool_payloads,
    tool_call,
)


@pytest.fixture
def mcp_url(settings, services):
    port = free_port()
    settings.mcp_base_url = f"http://127.0.0.1:{port}/mcp"
    shutil.copy(SAMPLES / "synthetic_gnn_experiments.csv", settings.experiments_dir)
    services.store.ingest((SAMPLES / "gnn_primer.pdf").read_bytes(), "gnn_primer.pdf")
    app = build_mcp_app(create_mcp_server(services), settings)
    with ServerThread(app, port):
        yield settings.mcp_base_url


def client(url, token=MCP_TOKEN):
    return Client(StreamableHttpTransport(url, auth=token), timeout=30)


async def call(url, name, args):
    async with client(url) as c:
        return await c.call_tool(name, args, raise_on_error=False)


async def test_tool_registration_and_discovery(mcp_url):
    async with client(mcp_url) as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert sorted(tools) == sorted(TOOL_NAMES)
    assert tools["search_papers"].input_schema["properties"]["max_results"]["maximum"] == 50
    assert tools["analyze_repository"].input_schema["required"] == ["repository"]
    assert tools["query_documents"].output_schema["properties"]["passages"]
    assert tools["generate_report"].annotations.read_only_hint is False
    assert all(tools[n].annotations.read_only_hint for n in TOOL_NAMES if n != "generate_report")


async def test_search_papers_over_mcp(mcp_url):
    result = await call(mcp_url, "search_papers", {"query": "graph neural networks", "max_results": 3})
    assert not result.is_error
    papers = result.structured_content["papers"]
    assert papers[0]["url"] == "https://arxiv.org/abs/1609.02907"
    assert json.loads(result.content[0].text)["papers"][0]["title"].startswith("Semi-Supervised")


async def test_search_papers_validation_errors(mcp_url):
    assert (await call(mcp_url, "search_papers", {})).is_error
    bad_date = await call(mcp_url, "search_papers", {"query": "x", "start_date": "yesterday"})
    assert bad_date.is_error and "YYYY-MM-DD" in bad_date.content[0].text
    too_many = await call(mcp_url, "search_papers", {"query": "x", "max_results": 500})
    assert too_many.is_error


async def test_analyze_repository_over_mcp(mcp_url):
    result = await call(mcp_url, "analyze_repository", {"repository": "octo/tiny-gnn"})
    assert not result.is_error and "PyTorch Geometric" in result.structured_content["ml_frameworks"]
    missing = await call(mcp_url, "analyze_repository", {"repository": "octo/missing"})
    assert missing.is_error and "not found" in missing.content[0].text


async def test_query_documents_over_mcp(mcp_url):
    result = await call(mcp_url, "query_documents", {"question": "What is over-smoothing in deep GNNs?"})
    data = result.structured_content
    assert data["evidence_available"] and data["passages"][0]["citation"] == "[gnn_primer.pdf, p. 3]"


async def test_fetch_experiments_over_mcp(mcp_url):
    result = await call(mcp_url, "fetch_experiments", {"metrics": ["val_loss"], "baseline": "GCN"})
    comp = result.structured_content["comparisons"][0]
    assert comp["direction"] == "lower" and comp["winner"] == "GAT"
    traversal = await call(mcp_url, "fetch_experiments", {"files": ["../../etc/passwd"]})
    assert traversal.is_error and "Invalid file name" in traversal.content[0].text


async def test_generate_report_over_mcp(mcp_url, settings):
    result = await call(
        mcp_url,
        "generate_report",
        {
            "title": "GNN report",
            "research_question": "Does attention help?",
            "arxiv_ids": ["1609.02907"],
            "document_question": "limitations of message passing",
            "repository": "octo/tiny-gnn",
            "include_experiments": True,
            "metrics": ["val_accuracy"],
        },
    )
    assert not result.is_error, result.content
    data = result.structured_content
    md = (settings.reports_dir / f"{data['report_id']}.md").read_text()
    assert "gnn_primer.pdf" in md and "octo/tiny-gnn" in md and "Semi-Supervised" in md
    assert data["html_url"].endswith(f"/api/reports/{data['report_id']}?format=html")
    empty = await call(mcp_url, "generate_report", {"title": "x y z", "research_question": "nothing"})
    assert empty.is_error and "at least one verified evidence source" in empty.content[0].text


async def test_mcp_requires_bearer_token(mcp_url):
    for token in (None, "wrong-token-123456"):
        with pytest.raises(Exception):  # noqa: B017 - transport raises an auth error
            async with client(mcp_url, token) as c:
                await c.list_tools()
    async with httpx.AsyncClient() as raw:
        r = await raw.post(
            mcp_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Accept": "application/json, text/event-stream"},
        )
    assert r.status_code == 401


async def test_mcp_rejects_foreign_host_and_origin(mcp_url):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    headers = {"Accept": "application/json, text/event-stream", "Authorization": f"Bearer {MCP_TOKEN}"}
    async with httpx.AsyncClient() as raw:
        bad_host = await raw.post(mcp_url, json=body, headers={**headers, "Host": "evil.example"})
        bad_origin = await raw.post(mcp_url, json=body, headers={**headers, "Origin": "https://evil.example"})
    assert bad_host.status_code == 421
    assert bad_origin.status_code == 403


async def test_langgraph_agent_calls_tools_through_mcp(mcp_url, settings):
    """The agent discovers tools from the MCP server and every tool call crosses the MCP transport."""

    def final(messages):
        payloads = last_tool_payloads(messages)
        papers = payloads["search_papers"]["papers"]
        comp = payloads["fetch_experiments"]["comparisons"][0]
        return AIMessage(
            content=f"[{papers[0]['title']}]({papers[0]['url']}); winner {comp['winner']}; "
            "see [gnn_primer.pdf, p. 3]"
        )

    model = ScriptedChatModel(
        steps=[
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("search_papers", {"query": "gnn", "max_results": 2}, "1"),
                    tool_call("fetch_experiments", {"metrics": ["val_accuracy"]}, "2"),
                    tool_call("query_documents", {"question": "over-smoothing"}, "3"),
                ],
            ),
            final,
        ]
    )
    provider = MCPToolProvider(mcp_url, MCP_TOKEN)
    result = await ResearchAgent(model, provider, settings).run("research GNNs")
    assert sorted(model.bound_tools) == sorted(TOOL_NAMES)
    assert [t["status"] for t in result.trace] == ["success"] * 3
    assert "winner GAT" in result.answer
    assert result.citation_check["unverified"] == []
    tools = await provider.get_tools()
    assert all(t.metadata["mcp"]["server"]["name"] == "mcp-research-agent" for t in tools)
