"""Deterministic fakes for third-party APIs and the LLM."""

from __future__ import annotations

import base64
import json
import socket
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.config import Settings

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
API_KEY = "test-api-key-0123456789"
MCP_TOKEN = "test-mcp-token-0123456789"


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_env": "test",
        "chroma_mode": "ephemeral",
        "embedding_provider": "hash",
        "data_dir": tmp_path / "data",
        "reports_dir": tmp_path / "reports",
        "arxiv_min_interval_s": 0,
        "allowed_hosts": "testserver,localhost,127.0.0.1",
        "allowed_origins": "http://localhost:8000",
        "app_api_key": API_KEY,
        "mcp_auth_token": MCP_TOKEN,
        "openai_api_key": None,
        "github_token": None,
        "rag_max_distance": 0.92,
        "rate_limit_per_minute": 1000,
        "chat_rate_limit_per_minute": 1000,
        "public_base_url": "http://testserver",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


# ---------------------------------------------------------------- arXiv


def atom_entry(
    arxiv_id: str,
    title: str,
    authors: Sequence[str] = ("Ada Lovelace", "Alan Turing"),
    summary: str = "We study graph neural networks. Results are reported.",
    published: str = "2023-05-01T00:00:00Z",
) -> str:
    author_xml = "".join(f"<author><name>{a}</name></author>" for a in authors)
    return f"""<entry><id>http://arxiv.org/abs/{arxiv_id}v1</id><updated>{published}</updated>
<published>{published}</published><title>{title}</title><summary>{summary}</summary>{author_xml}
<link href="http://arxiv.org/abs/{arxiv_id}v1" rel="alternate" type="text/html"/>
<link title="pdf" href="http://arxiv.org/pdf/{arxiv_id}v1" rel="related" type="application/pdf"/>
<category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/></entry>"""


def atom_feed(entries: Sequence[str], total: int | None = None) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
 xmlns:arxiv="http://arxiv.org/schemas/atom"><title>ArXiv Query</title>
<opensearch:totalResults>{len(entries) if total is None else total}</opensearch:totalResults>
<opensearch:startIndex>0</opensearch:startIndex>{"".join(entries)}</feed>"""


DEFAULT_PAPERS = {
    "1609.02907": "Semi-Supervised Classification with Graph Convolutional Networks",
    "1710.10903": "Graph Attention Networks",
    "1706.02216": "Inductive Representation Learning on Large Graphs",
}


def arxiv_handler(calls: list[httpx.Request] | None = None) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        params = request.url.params
        if "id_list" in params:
            ids = params["id_list"].split(",")
            entries = [atom_entry(i, DEFAULT_PAPERS.get(i, f"Paper {i}")) for i in ids if i in DEFAULT_PAPERS]
            return httpx.Response(200, text=atom_feed(entries))
        query = params.get("search_query", "")
        if "nothingmatches" in query:
            return httpx.Response(200, text=atom_feed([]))
        entries = [atom_entry(i, t) for i, t in DEFAULT_PAPERS.items()]
        return httpx.Response(200, text=atom_feed(entries, total=1234))

    return handler


# ---------------------------------------------------------------- GitHub


def _b64(text: str) -> dict[str, Any]:
    raw = text.encode()
    return {"encoding": "base64", "content": base64.b64encode(raw).decode(), "size": len(raw)}


FAKE_REPO_FILES = {
    "README.md": "# tiny-gnn\nA minimal GCN implementation in PyTorch Geometric.\n",
    "requirements.txt": "torch>=2.0\ntorch-geometric==2.5.0\nnumpy\n",
    "src/model.py": (
        "import torch\nfrom torch_geometric.nn import GCNConv\n\n\n"
        "class GCN(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n"
        "        self.conv1 = GCNConv(16, 16)\n\n    def forward(self, x, edge_index):\n"
        "        return self.conv1(x, edge_index)\n"
    ),
    "train.py": "from src.model import GCN\n\n\ndef train(model, data):\n    pass\n",
    ".env": "OPENAI_API_KEY=sk-should-never-be-fetched\n",
    "tests/test_model.py": "def test_x():\n    assert True\n",
    ".github/workflows/ci.yml": "name: ci\n",
}


def github_handler(
    calls: list[httpx.Request] | None = None, rate_limited: bool = False
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        path = request.url.path
        if rate_limited:
            return httpx.Response(
                403,
                json={"message": "API rate limit exceeded"},
                headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1900000000"},
            )
        headers = {"X-RateLimit-Remaining": "55"}
        base = "/repos/octo/tiny-gnn"
        if not path.startswith(base) and path.startswith("/repos/"):
            return httpx.Response(404, json={"message": "Not Found"}, headers=headers)
        if path == base:
            return httpx.Response(
                200,
                headers=headers,
                json={
                    "full_name": "octo/tiny-gnn",
                    "html_url": "https://github.com/octo/tiny-gnn",
                    "description": "Tiny GCN <script>alert(1)</script>",
                    "default_branch": "main",
                    "stargazers_count": 42,
                    "forks_count": 3,
                    "open_issues_count": 1,
                    "license": {"spdx_id": "MIT"},
                    "topics": ["gnn"],
                    "created_at": "2024-01-01T00:00:00Z",
                    "pushed_at": "2024-06-01T00:00:00Z",
                },
            )
        if path == f"{base}/languages":
            return httpx.Response(200, json={"Python": 9000, "Shell": 1000}, headers=headers)
        if path == f"{base}/readme":
            return httpx.Response(200, json=_b64(FAKE_REPO_FILES["README.md"]), headers=headers)
        if path == f"{base}/branches/main":
            return httpx.Response(200, json={"commit": {"sha": "a" * 40}}, headers=headers)
        if path.startswith(f"{base}/git/trees/"):
            tree = [{"path": p, "type": "blob", "size": len(c)} for p, c in FAKE_REPO_FILES.items()]
            tree += [{"path": "src", "type": "tree"}, {"path": "tests", "type": "tree"}]
            return httpx.Response(200, json={"tree": tree, "truncated": False}, headers=headers)
        if path.startswith(f"{base}/contents/"):
            file_path = path.removeprefix(f"{base}/contents/")
            if file_path in FAKE_REPO_FILES:
                return httpx.Response(
                    200, json={**_b64(FAKE_REPO_FILES[file_path]), "path": file_path}, headers=headers
                )
        return httpx.Response(404, json={"message": "Not Found"}, headers=headers)

    return handler


# ---------------------------------------------------------------- LLM

Step = AIMessage | Callable[[list[BaseMessage]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """Chat model that replays scripted responses (optionally computed from the conversation)."""

    steps: list[Any]
    calls: list[list[BaseMessage]] = []
    bound_tools: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [t.name for t in tools]
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        step = self.steps.pop(0) if self.steps else AIMessage(content="(script exhausted)")
        return step(messages) if callable(step) else step

    def _generate(self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None, **kw):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    async def _agenerate(
        self, messages, stop=None, run_manager: AsyncCallbackManagerForLLMRun | None = None, **kw
    ):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])


def tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def last_tool_payloads(messages: list[BaseMessage]) -> dict[str, Any]:
    out = {}
    for m in messages:
        if isinstance(m, ToolMessage):
            text = m.text if isinstance(m.content, list) else str(m.content)
            try:
                out[m.name] = json.loads(text)
            except ValueError:
                out[m.name] = text
    return out


# ---------------------------------------------------------------- server


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    def __init__(self, app: Any, port: int) -> None:
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> ServerThread:
        self.thread.start()
        deadline = time.time() + 15
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
