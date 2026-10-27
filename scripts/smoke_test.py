"""End-to-end smoke test against a running deployment (e.g. after `docker compose up --build -d`).

Checks: API reachable, ChromaDB reachable (via /readyz), MCP discovery of all five tools, an MCP tool invocation,
PDF indexing + MCP document query, experiment upload + MCP analysis, and report generation + download.
Uses no paid APIs: it calls MCP tools directly and never invokes the LLM.

    python scripts/smoke_test.py --api http://localhost:8000 --mcp http://localhost:8000/mcp
Reads APP_API_KEY and MCP_AUTH_TOKEN from the environment (or a .env file in the current directory).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
EXPECTED_TOOLS = {
    "search_papers",
    "analyze_repository",
    "query_documents",
    "fetch_experiments",
    "generate_report",
}


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Smoke:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}{' — ' + detail if detail else ''}")
        self.failures += 0 if ok else 1
        return ok


async def run(api: str, mcp_url: str, api_key: str | None, mcp_token: str | None, wait: int) -> int:
    s = Smoke()
    headers = {"X-API-Key": api_key} if api_key else {}
    async with httpx.AsyncClient(base_url=api, timeout=120, headers=headers) as http:
        deadline = time.time() + wait
        ready = None
        while time.time() < deadline:
            try:
                ready = await http.get("/readyz")
                if ready.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(3)
        if not s.check("API reachable (/healthz)", (await http.get("/healthz")).status_code == 200):
            return 1
        checks = ready.json().get("checks", {}) if ready is not None else {}
        s.check("ChromaDB reachable", checks.get("chromadb") == "ok", str(checks.get("chromadb")))
        s.check("Service ready (/readyz)", ready is not None and ready.status_code == 200, str(checks))

        mcp = Client(StreamableHttpTransport(mcp_url, auth=mcp_token), timeout=120)
        async with mcp:
            tools = {t.name for t in await mcp.list_tools()}
            s.check("MCP discovers all five tools", tools == EXPECTED_TOOLS, ", ".join(sorted(tools)))

            probe = await mcp.call_tool(
                "query_documents", {"question": "smoke test connectivity probe"}, raise_on_error=False
            )
            s.check(
                "Simple MCP tool invocation succeeds",
                not probe.is_error and probe.structured_content is not None,
                (probe.structured_content or {}).get("message", "")[:80],
            )

            pdf = (SAMPLES / "gnn_primer.pdf").read_bytes()
            r = await http.post("/api/upload/pdf", files={"file": ("gnn_primer.pdf", pdf, "application/pdf")})
            s.check("PDF upload indexed", r.status_code in (200, 201), r.text[:160])
            q = await mcp.call_tool(
                "query_documents",
                {"question": "What are the limitations of message passing, such as over-smoothing?"},
                raise_on_error=False,
            )
            data = q.structured_content or {}
            top = (data.get("passages") or [{}])[0]
            s.check(
                "Document query returns page citation",
                not q.is_error and data.get("evidence_available") is True and top.get("page") == 3,
                f"{top.get('citation')} score={top.get('score')}",
            )

            off = await mcp.call_tool(
                "query_documents",
                {"question": "What is the boiling point of olive oil in a frying pan?"},
                raise_on_error=False,
            )
            s.check(
                "Off-topic question reports evidence unavailable",
                not off.is_error and (off.structured_content or {}).get("evidence_available") is False,
            )

            csv = (SAMPLES / "synthetic_gnn_experiments.csv").read_bytes()
            r = await http.post(
                "/api/upload/experiment",
                files={"file": ("synthetic_gnn_experiments.csv", csv, "text/csv")},
            )
            s.check("Experiment CSV upload", r.status_code in (200, 201), r.text[:160])
            name = r.json().get("file_name", "synthetic_gnn_experiments.csv") if r.status_code < 300 else ""
            a = await mcp.call_tool(
                "fetch_experiments",
                {"files": [name], "metrics": ["val_accuracy", "val_loss"]},
                raise_on_error=False,
            )
            comps = (a.structured_content or {}).get("comparisons", [])
            summary = "; ".join(f"{c['metric']} ({c['direction']}) best={c['winner']}" for c in comps)
            s.check("Experiment analysis", not a.is_error and len(comps) == 2, summary)

            rep = await mcp.call_tool(
                "generate_report",
                {
                    "title": "Smoke test: GCN vs GAT (synthetic data)",
                    "research_question": "On the synthetic logs, does GAT outperform GCN?",
                    "document_question": "limitations of message passing",
                    "experiment_files": [name],
                    "metrics": ["val_accuracy", "val_loss"],
                },
                raise_on_error=False,
            )
            report = rep.structured_content or {}
            s.check(
                "Report generated",
                not rep.is_error and bool(report.get("report_id")),
                report.get("report_id", ""),
            )
            if report.get("report_id"):
                for fmt in ("md", "html"):
                    d = await http.get(f"/api/reports/{report['report_id']}", params={"format": fmt})
                    ok = d.status_code == 200 and "References" in d.text
                    s.check(f"Report download ({fmt})", ok, f"{len(d.content)} bytes")

        if api_key:
            unauth = await httpx.AsyncClient(base_url=api).get("/api/tools")
            s.check("API rejects missing key", unauth.status_code == 401)
    print(f"\n{'All checks passed' if not s.failures else f'{s.failures} check(s) failed'}.")
    return 1 if s.failures else 0


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api", default=os.environ.get("SMOKE_API_URL", "http://localhost:8000"))
    parser.add_argument("--mcp", default=os.environ.get("SMOKE_MCP_URL", "http://localhost:8000/mcp"))
    parser.add_argument("--wait", type=int, default=120, help="seconds to wait for readiness")
    args = parser.parse_args()
    code = asyncio.run(
        run(
            args.api.rstrip("/"),
            args.mcp,
            os.environ.get("APP_API_KEY") or None,
            os.environ.get("MCP_AUTH_TOKEN") or None,
            args.wait,
        )
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
