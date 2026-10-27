"""FastAPI endpoints against the full app served by uvicorn (agent → MCP over HTTP at /mcp)."""

import re

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.core.security import rate_limiter
from app.main import create_app
from scripts.make_samples import build_pdf
from tests.helpers import (
    API_KEY,
    SAMPLES,
    ScriptedChatModel,
    ServerThread,
    free_port,
    last_tool_payloads,
    tool_call,
)

AUTH = {"X-API-Key": API_KEY}
PDF = (SAMPLES / "gnn_primer.pdf").read_bytes()
CSV = (SAMPLES / "synthetic_gnn_experiments.csv").read_bytes()


def report_step(messages):
    payloads = last_tool_payloads(messages)
    if "generate_report" in payloads:
        r = payloads["generate_report"]
        return AIMessage(
            content=f"Report ready: [Markdown]({r['markdown_url']}) — cites "
            "https://arxiv.org/abs/1609.02907 and [gnn_primer.pdf, p. 3]"
        )
    return AIMessage(
        content="",
        tool_calls=[
            tool_call(
                "generate_report",
                {
                    "title": "GNN comparison",
                    "research_question": "Is GAT better than GCN?",
                    "arxiv_ids": ["1609.02907"],
                    "document_question": "over-smoothing",
                    "include_experiments": True,
                },
                "r1",
            )
        ],
    )


@pytest.fixture
def server(settings, services):
    port = free_port()
    settings.mcp_base_url = f"http://127.0.0.1:{port}/mcp"
    settings.public_base_url = f"http://127.0.0.1:{port}"
    services.reports.public_base_url = settings.public_base_url
    model = ScriptedChatModel(
        steps=[
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("search_papers", {"query": "graph neural networks"}, "s1"),
                    tool_call("fetch_experiments", {"metrics": ["val_accuracy"]}, "s2"),
                    tool_call("analyze_repository", {"repository": "octo/tiny-gnn"}, "s3"),
                ],
            ),
            AIMessage(
                content="GAT beats GCN on val_accuracy. See https://arxiv.org/abs/1609.02907 and "
                "https://arxiv.org/abs/2999.12345"
            ),
            report_step,
            report_step,
        ]
    )
    app = create_app(settings, services=services, model=model)
    with ServerThread(app, port), httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as client:
        client.model = model
        yield client


def test_health_and_readiness(server):
    assert server.get("/healthz").json() == {"status": "ok", "checks": {}}
    ready = server.get("/readyz")
    assert ready.status_code == 200, ready.text
    checks = ready.json()["checks"]
    assert checks["chromadb"] == "ok" and checks["mcp"] == "ok (5 tools)" and checks["llm"] == "configured"


def test_openapi_and_frontend(server):
    spec = server.get("/openapi.json").json()
    for path in [
        "/api/chat",
        "/api/upload/pdf",
        "/api/upload/experiment",
        "/api/reports/{report_id}",
        "/api/tools",
    ]:
        assert path in spec["paths"]
    page = server.get("/")
    assert page.status_code == 200 and "MCP Research Agent" in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert server.get("/api/config").json()["auth_required"] is True


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/api/chat"),
        ("POST", "/api/upload/pdf"),
        ("POST", "/api/upload/experiment"),
        ("GET", "/api/reports/" + "a" * 32),
        ("GET", "/api/tools"),
        ("GET", "/api/documents"),
    ],
)
def test_endpoints_require_api_key(server, method, path):
    assert server.request(method, path).status_code == 401
    assert server.request(method, path, headers={"X-API-Key": "wrong"}).status_code == 401


def test_bearer_auth_is_accepted(server):
    assert server.get("/api/tools", headers={"Authorization": f"Bearer {API_KEY}"}).status_code == 200


def test_mcp_endpoint_mounted_and_protected(server):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    accept = {"Accept": "application/json, text/event-stream"}
    assert server.post("/mcp", json=body, headers=accept).status_code == 401
    assert (
        server.post("/mcp", json=body, headers={**accept, "Authorization": f"Bearer {API_KEY}"}).status_code
        == 401
    )  # the REST API key is not an MCP token


def test_tools_endpoint_lists_mcp_tools(server):
    data = server.get("/api/tools", headers=AUTH).json()
    assert data["mcp_url"].endswith("/mcp")
    assert {t["name"] for t in data["tools"]} == {
        "search_papers",
        "analyze_repository",
        "query_documents",
        "fetch_experiments",
        "generate_report",
    }


def test_pdf_upload_validation(server):
    files = {"file": ("gnn primer.pdf", PDF, "application/pdf")}
    first = server.post("/api/upload/pdf", files=files, headers=AUTH)
    assert first.status_code == 201, first.text
    assert first.json()["pages"] == 3 and first.json()["filename"] == "gnn_primer.pdf"
    dup = server.post("/api/upload/pdf", files=files, headers=AUTH)
    assert dup.status_code == 200 and dup.json()["duplicate"] is True
    assert server.get("/api/documents", headers=AUTH).json()[0]["document_id"] == first.json()["document_id"]

    cases = [
        (("notes.txt", b"hello", "text/plain"), 415),
        (("fake.pdf", b"just text", "application/pdf"), 422),
        (("broken.pdf", b"%PDF-1.4\nnot really", "application/pdf"), 422),
        (("scan.pdf", build_pdf([("", [])], "scan").replace(b"() Tj", b""), "application/pdf"), 422),
        (("evil.pdf", PDF, "text/html"), 415),
        (("big.pdf", b"%PDF-" + b"0" * (21 * 1024 * 1024), "application/pdf"), 413),
    ]
    for file, code in cases:
        r = server.post("/api/upload/pdf", files={"file": file}, headers=AUTH)
        assert r.status_code == code, (file[0], r.status_code, r.text)
    scan = server.post("/api/upload/pdf", files={"file": cases[3][0]}, headers=AUTH)
    assert "image-only" in scan.json()["detail"]


def test_experiment_upload_validation(server, settings):
    r = server.post(
        "/api/upload/experiment", files={"file": ("../../../tmp/x y.csv", CSV, "text/csv")}, headers=AUTH
    )
    assert r.status_code == 201, r.text
    assert r.json()["file_name"] == "x_y.csv" and "val_accuracy" in r.json()["metric_columns"]
    assert (settings.experiments_dir / "x_y.csv").exists()
    assert (
        server.post(
            "/api/upload/experiment", files={"file": ("x y.csv", CSV, "text/csv")}, headers=AUTH
        ).json()["duplicate"]
        is True
    )
    other = server.post(
        "/api/upload/experiment",
        files={"file": ("x y.csv", CSV + b"GCN-x,GCN,d,9,1,1,1,1,1\n", "text/csv")},
        headers=AUTH,
    )
    assert other.status_code == 201 and re.fullmatch(r"x_y-[a-f0-9]{8}\.csv", other.json()["file_name"])
    for file, code in [
        (("run.py", b"import os", "text/x-python"), 415),
        (("bad.json", b"{oops", "application/json"), 422),
        (("words.csv", b"name,comment\na,b\n", "text/csv"), 422),
    ]:
        assert server.post("/api/upload/experiment", files={"file": file}, headers=AUTH).status_code == code
    assert server.get("/api/experiments", headers=AUTH).json() == sorted(
        [p.name for p in settings.experiments_dir.iterdir() if not p.name.startswith(".")]
    )


def test_chat_multi_tool_report_and_download(server):
    server.post("/api/upload/pdf", files={"file": ("gnn_primer.pdf", PDF, "application/pdf")}, headers=AUTH)
    server.post("/api/upload/experiment", files={"file": ("runs.csv", CSV, "text/csv")}, headers=AUTH)

    r = server.post("/api/chat", json={"message": "Compare GCN and GAT"}, headers=AUTH)
    assert r.status_code == 200, r.text
    data = r.json()
    assert [t["tool"] for t in data["tool_trace"]] == [
        "search_papers",
        "fetch_experiments",
        "analyze_repository",
    ]
    assert all(t["status"] == "success" for t in data["tool_trace"])
    assert data["experiment_tables"][0]["comparisons"][0]["winner"] == "GAT"
    types = {c["type"] for c in data["citations"]}
    assert {"paper", "repository", "code", "experiment_file"} <= types
    assert data["citation_check"]["unverified"] == ["arXiv:2999.12345"]
    assert "Citation check" in data["answer"]

    # second turn on the same thread: report generation
    r2 = server.post(
        "/api/chat", json={"message": "Write a report", "thread_id": data["thread_id"]}, headers=AUTH
    )
    assert r2.status_code == 200, r2.text
    report = r2.json()["reports"][0]
    assert r2.json()["citation_check"]["unverified"] == []
    md_path = httpx.URL(report["markdown_url"]).raw_path.decode()
    md = server.get(md_path, headers=AUTH)
    assert md.status_code == 200 and md.headers["content-type"].startswith("text/markdown")
    assert "## References" in md.text and "attachment" in md.headers["content-disposition"]
    html = server.get(md_path.replace("format=md", "format=html"), headers=AUTH)
    assert html.status_code == 200 and "default-src 'none'" in html.headers["content-security-policy"]
    assert server.get(md_path).status_code == 401
    # the multi-turn history reached the model
    last_call = server.model.calls[-1]
    assert sum(1 for m in last_call if m.type == "human") == 2


def test_report_download_rejects_bad_ids(server):
    assert server.get("/api/reports/" + "0" * 32, headers=AUTH).status_code == 404
    assert server.get("/api/reports/..%2F..%2Fetc%2Fpasswd", headers=AUTH).status_code == 404
    assert server.get("/api/reports/" + "a" * 32 + "?format=exe", headers=AUTH).status_code == 422


def test_chat_validation(server):
    assert server.post("/api/chat", json={"message": ""}, headers=AUTH).status_code == 422
    assert server.post("/api/chat", json={"message": "x" * 4001}, headers=AUTH).status_code == 422
    assert (
        server.post("/api/chat", json={"message": "hi", "thread_id": "../x"}, headers=AUTH).status_code == 422
    )


def test_untrusted_host_rejected(server):
    assert server.get("/healthz", headers={"Host": "evil.example"}).status_code == 400


def test_rate_limit(server, settings):
    settings.rate_limit_per_minute = 3
    rate_limiter._hits.clear()
    codes = [server.get("/api/experiments", headers=AUTH).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    settings.rate_limit_per_minute = 1000
    rate_limiter._hits.clear()


def test_chat_without_llm_returns_503(settings, services):
    port = free_port()
    settings.mcp_base_url = f"http://127.0.0.1:{port}/mcp"
    with ServerThread(create_app(settings, services=services, model=None), port):
        r = httpx.post(f"http://127.0.0.1:{port}/api/chat", json={"message": "hi"}, headers=AUTH)
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]
