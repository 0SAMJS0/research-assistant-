"""Run the MCP tool server standalone: `python -m app.mcp_server` (Streamable HTTP at /mcp)."""

from __future__ import annotations

import logging
import os

import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.mcp_server.server import ToolServices, build_mcp_app, create_mcp_server


def build_app():
    settings = get_settings()
    settings.ensure_dirs()
    if not settings.secret("mcp_auth_token"):
        logging.getLogger(__name__).warning(
            "MCP_AUTH_TOKEN is not set: MCP tool calls are unauthenticated (development only)."
        )
    mcp = create_mcp_server(ToolServices.from_settings(settings), settings.mcp_base_url)

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "mcp"})

    return build_mcp_app(mcp, settings)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        build_app(),
        host=os.environ.get("MCP_HOST", "0.0.0.0"),  # noqa: S104 - container bind
        port=int(os.environ.get("MCP_PORT", "8100")),
        log_level=settings.log_level.lower(),
        proxy_headers=True,
        server_header=False,
    )


if __name__ == "__main__":
    main()
