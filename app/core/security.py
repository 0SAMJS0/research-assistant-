"""Authentication, rate limiting and safe-path helpers shared by the API and MCP server."""

from __future__ import annotations

import hmac
import re
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from mcp.server.auth.provider import AccessToken

from app.core.config import Settings

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REPORT_ID_RE = re.compile(r"^[a-f0-9]{32}$")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer = HTTPBearer(auto_error=False)


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


async def require_api_key(
    request: Request,
    api_key: str | None = Depends(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Require APP_API_KEY via `X-API-Key` or `Authorization: Bearer`.

    Development without APP_API_KEY runs open (a warning is logged at startup);
    production refuses to start without it (see Settings validation).
    """
    settings: Settings = request.app.state.container.settings
    expected = settings.secret("app_api_key")
    if not expected:
        return
    supplied = api_key or (bearer.credentials if bearer else None)
    if not supplied or not constant_time_equals(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def request_has_valid_key(request: Request, settings: Settings) -> bool:
    """Header check used by middleware so unauthenticated requests are rejected before body parsing."""
    expected = settings.secret("app_api_key")
    if not expected:
        return True
    supplied = request.headers.get("x-api-key")
    auth = request.headers.get("authorization", "")
    if not supplied and auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    return bool(supplied) and constant_time_equals(supplied, expected)


class StaticTokenVerifier:
    """MCP SDK TokenVerifier that accepts a single shared bearer token (MCP_AUTH_TOKEN)."""

    def __init__(self, token: str, resource: str | None = None) -> None:
        self._token = token
        self._resource = resource

    async def verify_token(self, token: str) -> AccessToken | None:
        if not constant_time_equals(token, self._token):
            return None
        return AccessToken(token=token, client_id="mcp-research-agent", scopes=[], resource=self._resource)


class RateLimiter:
    """In-memory sliding-window limiter keyed by client and bucket (single-process scope)."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def check(self, key: str, bucket: str, limit: int, window_s: float = 60.0) -> bool:
        now = time.monotonic()
        hits = self._hits[(key, bucket)]
        while hits and now - hits[0] > window_s:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        return True


rate_limiter = RateLimiter()


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def safe_child(base: Path, name: str) -> Path:
    """Resolve `name` as a direct child of `base`, rejecting traversal and odd names."""
    if not SAFE_NAME_RE.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"Invalid file name: {name!r}")
    base_resolved = base.resolve()
    candidate = (base_resolved / name).resolve()
    if candidate.parent != base_resolved:
        raise ValueError(f"Invalid file name: {name!r}")
    return candidate


def sanitize_filename(name: str, default: str = "upload") -> str:
    stem = Path(name or default).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (cleaned or default)[:100]
