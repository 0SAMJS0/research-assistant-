# syntax=docker/dockerfile:1.7
# One image runs both the API (default CMD) and the standalone MCP server (`python -m app.mcp_server`).

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.27 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH" HOME=/home/app \
    DATA_DIR=/data/app REPORTS_DIR=/data/reports
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app \
    && mkdir -p /data/app/experiments /data/reports \
    && chown -R app:app /data
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app app ./app
COPY --chown=app:app frontend ./frontend
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app data/samples ./data/samples
USER app
# Bake the local embedding model (all-MiniLM-L6-v2 ONNX, ~80 MB) into the image so runtime needs no download.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warm-up'])"
VOLUME ["/data"]
EXPOSE 8000 8100
HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import os,urllib.request,sys; p=os.environ.get('HEALTH_PORT','8000'); \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/healthz',timeout=4).status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", \
     "--forwarded-allow-ips", "*", "--no-server-header"]
