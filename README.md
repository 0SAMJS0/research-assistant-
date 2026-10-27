# MCP Research Agent

An AI research assistant whose LangGraph agent works exclusively through **real Model Context Protocol
tools** served over Streamable HTTP. It searches arXiv, analyzes GitHub repositories read-only, answers
questions from your PDFs with page-level citations (ChromaDB RAG), compares ML experiment logs with pandas, and
writes cited Markdown/HTML research reports, all from a small web UI and an authenticated REST API.

```
Browser UI ──► FastAPI (/api/*) ──► LangGraph agent ──MCP client (Streamable HTTP, bearer auth)──► MCP server (/mcp)
                     │                                                                  │
                     └── uploads ──► ChromaDB server ◄── query_documents ───────────────┤
                                     data volume     ◄── fetch_experiments / reports ───┤
                                                         arXiv API / GitHub REST API ◄──┘
```

## Features

- **Five MCP tools** registered with the MCP Python SDK v2 `MCPServer` (typed input *and* output schemas, tool
  annotations), served over Streamable HTTP at `/mcp` with bearer-token auth and Host/Origin
  (DNS-rebinding) protection.
- **LangGraph agent**: an explicit `StateGraph` (typed state, `agent ⇄ tools` loop) that discovers tools from the
  MCP server through LangChain's official `langchain.mcp.MCPAdapter`. Every tool call crosses the MCP transport.
  Includes a per-request tool-call budget, recursion limit, per-tool and per-request timeouts, failed-tool
  recovery, multi-turn memory by `thread_id`, and an execution trace (tool, status, duration).
- **Citation grounding**: after each answer, every `[file.pdf, p. N]`, arXiv ID, GitHub repo and report link is
  checked against what the tools actually returned in that conversation. Anything else is flagged in the answer.
- **RAG**: page-aware PDF extraction (pypdf), configurable chunk size and overlap, SHA-256 duplicate detection,
  embeddings in a server-backed ChromaDB, cosine top-k with a relevance cut-off, and "evidence unavailable"
  answers when nothing relevant is found. Empty, malformed, encrypted and image-only PDFs are rejected with clear
  errors.
- **Experiment analysis**: CSV/JSON (flat or nested `history`) logs, automatic metric detection, filters
  (model/seed/dataset), per-run final values at the last epoch, mean ± sample std across seeds, best observed
  value, and percentage improvement that accounts for whether lower or higher is better.
- **Reports**: the report tool re-fetches all evidence itself (arXiv metadata, PDF passages, repository analysis,
  experiment statistics), so it cannot accept invented facts. It writes Markdown + sanitized HTML with numbered
  references, labels observed facts separately from model interpretation, and strips unverified links from that
  interpretation.
- **Security**: API-key auth (fail-closed in production), MCP bearer auth, upload type/size/content validation,
  path-traversal-proof storage, outbound HTTPS allow-list (SSRF guard, redirects re-checked), timeouts, rate
  limits, CORS, trusted hosts, CSP, secret-redacting logs, non-root read-only containers, and a private Chroma.

## The five MCP tools

| Tool | What it does |
|---|---|
| `search_papers` | arXiv API search with max results, date range and sort; `related_to` finds papers related to an arXiv ID. Retries on 429/5xx with backoff, honours arXiv's 3 s rate limit, and caches results. |
| `analyze_repository` | GitHub REST API: metadata, languages, README, recursive tree, dependency manifests, selected source files. Detects ML frameworks and algorithm keywords and returns line-level code links. Never clones or runs code, and skips secret-looking files. A token is optional. |
| `query_documents` | Semantic search over uploaded PDFs. Returns passages with `[filename, p. N]` citation keys, or `evidence_available: false`. |
| `fetch_experiments` | Direction-aware comparison of uploaded experiment logs, read only from the approved data directory. |
| `generate_report` | Builds a cited Markdown and HTML report under a random 128-bit ID and returns download URLs. |

## Quick start (Docker, recommended)

Requires Docker with Compose v2.

```bash
cp .env.example .env
```

Edit `.env`: set `APP_API_KEY` and `MCP_AUTH_TOKEN` to random values (`openssl rand -hex 32`) and add
`OPENAI_API_KEY` to enable chat. Then start the stack:

```bash
docker compose up --build -d
```

```bash
docker compose ps
```

| Service | Address | Exposure |
|---|---|---|
| Web UI | http://localhost:8000/ | published on `127.0.0.1:8000` by default |
| REST API + OpenAPI docs | http://localhost:8000/docs | same |
| MCP endpoint (for external MCP clients) | http://localhost:8000/mcp | same, needs `Authorization: Bearer $MCP_AUTH_TOKEN` |
| Standalone MCP service (used by the agent) | http://mcp:8100/mcp | internal Docker network only |
| ChromaDB | http://chroma:8000 | internal Docker network only |

Open the UI, click **API key**, and paste your `APP_API_KEY`. Then run the smoke test:

```bash
uv run --no-dev python scripts/smoke_test.py --api http://localhost:8000 --mcp http://localhost:8000/mcp
```

Data persists in the `chroma-data` and `app-data` named volumes across `docker compose down` / `up`. Only
`docker compose down -v` deletes it.

## Local development (without Docker for the app)

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Start ChromaDB on port 8001:

```bash
docker run -d --name chroma -p 127.0.0.1:8001:8000 -v chroma-dev:/data chromadb/chroma:1.5.9
```

Create a development `.env`: copy `.env.example`, then set `APP_ENV=development`. Empty `APP_API_KEY` and
`MCP_AUTH_TOKEN` disable auth in development only, and a warning is logged.

```bash
cp .env.example .env
```

```bash
uv run uvicorn app.main:app --reload --port 8000
```

In this mode the agent calls the MCP endpoint mounted on the same app (`MCP_BASE_URL=http://localhost:8000/mcp`).
To run the MCP server on its own:

```bash
uv run python -m app.mcp_server
```

That server listens on `:8100/mcp`; point `MCP_BASE_URL` at it.

Regenerate the sample data:

```bash
uv run python scripts/make_samples.py
```

## Tests and lint

```bash
uv run ruff check .
```

```bash
uv run ruff format --check .
```

```bash
uv run pytest
```

The default suite is offline and deterministic. arXiv and GitHub are mocked with `httpx.MockTransport`, the LLM
is a scripted chat model, Chroma runs ephemeral with hashing embeddings, and the MCP server runs under uvicorn on
localhost, so the MCP tests use the real Streamable HTTP transport.

The optional live tests call real APIs:

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/integration/test_live_apis.py -v
```

| Live test | Needs |
|---|---|
| arXiv | outbound HTTPS |
| GitHub | outbound HTTPS; `GITHUB_TOKEN` optional |
| Agent end-to-end | `OPENAI_API_KEY` (paid) |

## REST API

All `/api/*` endpoints except `/api/config` require `X-API-Key: <APP_API_KEY>` or
`Authorization: Bearer <APP_API_KEY>`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | liveness |
| GET | `/readyz` | readiness: ChromaDB heartbeat, MCP tool discovery, writable storage, LLM status |
| POST | `/api/chat` | `{message, thread_id?}` → answer, `tool_trace`, `citations`, `experiment_tables`, `reports`, `citation_check` |
| POST | `/api/upload/pdf` | multipart `file` (PDF ≤ `MAX_UPLOAD_MB`); 201 indexed, 200 duplicate, 413/415/422 on errors |
| POST | `/api/upload/experiment` | multipart `file` (.csv/.json) |
| GET | `/api/reports/{report_id}?format=md\|html` | download a report |
| GET | `/api/tools` | tools as advertised by the MCP server |
| GET | `/api/documents`, `/api/experiments` | uploaded data |
| POST | `/mcp` | MCP Streamable HTTP endpoint (bearer `MCP_AUTH_TOKEN`) |

Example call:

```bash
curl -s localhost:8000/api/chat -H "X-API-Key: $APP_API_KEY" -H 'Content-Type: application/json' -d '{"message":"Find 2024 papers on graph transformers"}'
```

## Configuration

All settings are environment variables, documented in [.env.example](.env.example).

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `development` | `production` fails closed without strong `APP_API_KEY` and `MCP_AUTH_TOKEN` |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | none, `gpt-5-mini` | chat is disabled (HTTP 503) without a key; tools still work |
| `EMBEDDING_PROVIDER` | `local` | `local` ONNX MiniLM (baked into the image), `openai`, or `hash` (tests) |
| `GITHUB_TOKEN` | none | optional; raises the GitHub quota |
| `APP_API_KEY`, `MCP_AUTH_TOKEN` | none | REST API key and MCP bearer token |
| `PUBLIC_BASE_URL` | `http://localhost:8000` | used in report links |
| `MCP_BASE_URL` | `http://localhost:8000/mcp` | where the agent connects |
| `CHROMA_HOST`, `CHROMA_PORT` | `localhost`, `8001` | compose uses `chroma:8000` |
| `DATA_DIR`, `REPORTS_DIR` | `./data/runtime`, `./data/reports` | compose uses `/data/app`, `/data/reports` |
| `ALLOWED_ORIGINS`, `ALLOWED_HOSTS` | localhost | CORS and Host allow-lists (also used by MCP DNS-rebinding protection) |
| `MAX_UPLOAD_MB` | `20` | upload size cap |
| `AGENT_MAX_TOOL_CALLS`, `AGENT_RECURSION_LIMIT`, `AGENT_REQUEST_TIMEOUT_S`, `AGENT_TOOL_TIMEOUT_S` | 8, 25, 180, 90 | agent limits |

## Production deployment

See [docs/deployment.md](docs/deployment.md) for a Docker host deployment behind Caddy with automatic HTTPS
(`deploy/docker-compose.prod.yml`, `deploy/Caddyfile`). It also has an nginx alternative and covers backups and
upgrades. [docs/architecture.md](docs/architecture.md) and [docs/security.md](docs/security.md) describe the
design and the threat model.

## Demo questions

- "Find recent papers on graph neural networks for molecular property prediction published since 2024."
- "Find papers related to arXiv 1609.02907."
- "Analyze the repository pyg-team/pytorch_geometric: which frameworks and key algorithms does it use?"
- Upload `data/samples/gnn_primer.pdf`, then ask: "According to my PDF, what are the limitations of message passing?"
- Upload `data/samples/synthetic_gnn_experiments.csv`, then ask: "Compare GCN and GAT on validation accuracy and validation loss."
- "Write a cited report comparing my GCN and GAT experiments, with related arXiv work on graph attention."

## Sample data

Everything in `data/samples/` is **synthetic** and produced by `scripts/make_samples.py`. The experiment curves
come from a seeded random process and are not benchmark results. The PDF is original explanatory text written
for this project (MIT).

## Known limitations

- Conversation memory uses LangGraph's in-memory checkpointer, so threads are lost when the API restarts.
  Indexed documents, uploads and reports persist.
- Rate limiting is in-process, so per-replica limits apply if you run several API replicas.
- No OCR: scanned or image-only PDFs are rejected.
- Repository analysis is static pattern matching over at most 8 source files plus manifests.
- One shared API key and one MCP bearer token; there are no per-user accounts.

## License

MIT
