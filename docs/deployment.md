# Deployment guide

The stack is four containers on one private Docker network:

| Container | Role | Published |
|---|---|---|
| `api` | FastAPI REST API, web UI, and an MCP endpoint at `/mcp` for external MCP clients | via reverse proxy only |
| `mcp` | Standalone MCP tool server (`python -m app.mcp_server`, `:8100/mcp`), which the agent calls | no |
| `chroma` | ChromaDB server (vector store), persisted to the `chroma-data` volume | no |
| `caddy` | TLS termination and automatic HTTPS (production override only) | 80/443 |

`api` and `mcp` share the `app-data` volume (`/data/app` for uploads, `/data/reports` for generated reports). Both
run as UID 10001 with a read-only root filesystem, `no-new-privileges`, and all capabilities dropped.

## 1. Prepare a Docker host

Any Linux VM with Docker Engine 24+ and Compose v2 works (for example AWS EC2, GCP Compute Engine, Azure VM,
DigitalOcean, or Hetzner). Use at least 2 vCPU and 4 GB RAM; the image bundles a small ONNX embedding model.

1. Point a DNS `A`/`AAAA` record (e.g. `research.example.com`) at the host.
2. Open inbound TCP 80 and 443 (plus UDP 443 for HTTP/3) in the firewall. Do **not** open 8000, 8100 or the
   Chroma port.
3. Copy the repository to the host.

## 2. Configure

```bash
cp .env.example .env
```

```bash
chmod 600 .env
```

Set at least:

```dotenv
APP_ENV=production
APP_API_KEY=<openssl rand -hex 32>
MCP_AUTH_TOKEN=<openssl rand -hex 32>
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-mini
PUBLIC_BASE_URL=https://research.example.com
ALLOWED_HOSTS=research.example.com
ALLOWED_ORIGINS=https://research.example.com
GITHUB_TOKEN=            # optional, fine-grained token with no repository permissions
```

With `APP_ENV=production` the services refuse to start if `APP_API_KEY` or `MCP_AUTH_TOKEN` is missing, shorter
than 16 characters, or still the `change-me…` placeholder (fail closed).

## 3. Start with HTTPS

```bash
DOMAIN=research.example.com docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build
```

```bash
docker compose ps
```

Caddy obtains and renews Let's Encrypt certificates automatically. The override removes the API's host port, so
only Caddy is reachable from outside.

## 4. Verify

```bash
python scripts/smoke_test.py --api https://research.example.com --mcp https://research.example.com/mcp
```

The smoke test reads `APP_API_KEY` and `MCP_AUTH_TOKEN` from `.env` or the environment. It uploads the bundled
synthetic sample PDF and CSV. Run it on staging, or delete the samples afterwards if you don't want them in
production.

## Using nginx instead of Caddy

Keep `API_BIND=127.0.0.1` so the API is published on loopback only, start the base stack, and use
`deploy/nginx.conf`. Certificates come from certbot or your own CA.

```bash
docker compose up -d --build
```

The `/mcp` location disables proxy buffering so MCP server-sent events stream correctly.

## Connecting external MCP clients

Any MCP client that supports Streamable HTTP can use the tools:

- URL: `https://research.example.com/mcp`
- Header: `Authorization: Bearer <MCP_AUTH_TOKEN>`

The server validates the `Host` and `Origin` headers against `ALLOWED_HOSTS` and `ALLOWED_ORIGINS`
(DNS-rebinding protection), and it advertises OAuth protected-resource metadata at
`/.well-known/oauth-protected-resource/mcp`.

## Persistence, backups and upgrades

- Data lives in the named volumes `chroma-data` (vectors) and `app-data` (uploads and reports). Both survive
  `docker compose down`, `up`, restarts and image rebuilds. Only `docker compose down -v` deletes them.
- Backup sketch:

  ```bash
  docker run --rm -v mcp-research-agent_chroma-data:/v -v "$PWD":/b alpine tar czf /b/chroma-$(date +%F).tgz -C /v .
  ```

  Repeat for `mcp-research-agent_app-data`. Stop the stack, or at least `chroma`, for a consistent snapshot.
- Upgrade:

  ```bash
  git pull && docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build
  ```

- Changing `EMBEDDING_PROVIDER` or `OPENAI_EMBEDDING_MODEL` switches to a new Chroma collection, so vectors from
  different models never mix. Re-upload PDFs after switching.

## Managed platforms

The image is a standard OCI image (`docker build -t mcp-research-agent .`). On platforms that run one container
per service (Cloud Run, ECS/Fargate, Azure Container Apps, Fly.io, Render):

- Run the same image twice: once with the default command (API) and once with `python -m app.mcp_server`
  (MCP, port 8100). Set `MCP_BASE_URL` on the API to the MCP service's **private** URL.
- Run ChromaDB (`chromadb/chroma:1.5.9`) as a private service with a persistent disk mounted at `/data`. Never
  give it a public ingress.
- Mount a shared persistent volume (EFS, Azure Files, Fly volume, or similar) at `/data` on both app services, or
  run both processes in one service.
- Set `HEALTH_PORT` to the service port so the image's `HEALTHCHECK` probes the right process.

## Operations notes

- Logs go to stdout. API keys and tokens are redacted by a logging filter, and httpx request logging is lowered to
  WARNING.
- The in-memory rate limiter and conversation memory are per process. Run one API replica, or put a shared
  limiter and checkpointer in front before scaling horizontally.
