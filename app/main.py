"""FastAPI application: REST API + web UI, with the MCP Streamable HTTP endpoint mounted at /mcp."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models import BaseChatModel
from starlette.types import ASGIApp, Receive, Scope, Send

from app.agent.graph import ResearchAgent, build_chat_model
from app.agent.mcp_client import MCPToolProvider
from app.api.container import Container
from app.api.routes import health_router, public_router, router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.security import client_key, rate_limiter, request_has_valid_key
from app.mcp_server.server import ToolServices, build_mcp_app, create_mcp_server

logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MCP_PATHS = ("/mcp", "/.well-known/oauth-protected-resource")


class MCPDispatcher:
    """Routes /mcp (and its OAuth resource metadata) to the MCP ASGI app, everything else to FastAPI."""

    def __init__(self, api: FastAPI, mcp_app: ASGIApp) -> None:
        self.api = api
        self.mcp_app = mcp_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope["type"] == "http" and any(path == p or path.startswith(p + "/") for p in MCP_PATHS):
            await self.mcp_app(scope, receive, send)
        else:
            await self.api(scope, receive, send)


def create_app(
    settings: Settings | None = None,
    *,
    services: ToolServices | None = None,
    model: BaseChatModel | None = None,
) -> MCPDispatcher:
    """Build the ASGI app. `services` and `model` allow injecting mocked integrations / a fake LLM in tests."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()
    if not settings.secret("app_api_key"):
        logger.warning("APP_API_KEY is not set: REST API authentication is DISABLED (development only).")
    if not settings.secret("mcp_auth_token"):
        logger.warning("MCP_AUTH_TOKEN is not set: the /mcp endpoint is unauthenticated (development only).")

    services = services or ToolServices.from_settings(settings)
    mcp = create_mcp_server(services, resource_url=settings.public_base_url.rstrip("/") + "/mcp")
    mcp_app = build_mcp_app(mcp, settings)
    provider = MCPToolProvider(
        settings.mcp_base_url, settings.secret("mcp_auth_token"), settings.agent_tool_timeout_s
    )
    agent = ResearchAgent(model or build_chat_model(settings), provider, settings)
    container = Container(
        settings=settings, store=services.store, reports=services.reports, tool_provider=provider, agent=agent
    )

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        async with mcp.session_manager.run():
            yield

    api = FastAPI(
        title="MCP Research Agent",
        version="1.0.0",
        description="Research assistant API. The agent calls tools exclusively through the MCP server at /mcp. "
        "Authenticate with `X-API-Key` or `Authorization: Bearer <APP_API_KEY>`.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    api.state.container = container
    api.include_router(health_router)
    api.include_router(public_router)
    api.include_router(router)

    @api.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/"):
            if (
                path != "/api/config"
                and request.method != "OPTIONS"
                and not request_has_valid_key(request, settings)
            ):
                return JSONResponse(
                    {"detail": "Missing or invalid API key"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
            bucket, limit = (
                ("chat", settings.chat_rate_limit_per_minute)
                if path == "/api/chat"
                else ("api", settings.rate_limit_per_minute)
            )
            if not rate_limiter.check(client_key(request), bucket, limit):
                return JSONResponse(
                    {"detail": "Rate limit exceeded; try again shortly."},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        # UI assets revalidate on every load (so redeploys apply); API data is private and never cached.
        response.headers.setdefault("Cache-Control", "no-store" if path.startswith("/api/") else "no-cache")
        if not path.startswith(("/api/reports/", "/docs", "/openapi.json")):
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            )
        return response

    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Authorization", "Content-Type"],
    )
    api.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.hosts or ["localhost"])
    if FRONTEND_DIR.is_dir():
        api.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    dispatcher = MCPDispatcher(api, mcp_app)
    dispatcher.state = api.state  # type: ignore[attr-defined]
    return dispatcher


def __getattr__(name: str):
    # Lazily build `app` so importing this module (e.g. in tests) does not require a configured environment.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
