"""Outbound HTTP client restricted to an allow-list of public API hosts (SSRF protection)."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

USER_AGENT = "mcp-research-agent/1.0 (+https://github.com/; research assistant)"


class ExternalServiceError(RuntimeError):
    """Raised for failures talking to an external API; message is safe to show to users."""


def build_client(
    base_url: str,
    allowed_hosts: set[str],
    timeout_s: float,
    headers: dict[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError(f"Refusing outbound base URL {base_url!r}: only https to {sorted(allowed_hosts)}")

    async def _guard(request: httpx.Request) -> None:
        # Redirects and absolute URLs are re-checked so the client cannot be steered elsewhere.
        if request.url.scheme != "https" or request.url.host not in allowed_hosts:
            raise ExternalServiceError(f"Blocked request to non-allow-listed host {request.url.host!r}")

    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout_s),
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        follow_redirects=True,
        max_redirects=3,
        event_hooks={"request": [_guard]},
        limits=httpx.Limits(max_connections=10),
        transport=transport,
    )
