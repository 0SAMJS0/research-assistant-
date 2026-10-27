"""MCP client used by the agent: discovers tools from the MCP server over Streamable HTTP."""

from __future__ import annotations

import asyncio
import logging
import warnings

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain_core._api import LangChainBetaWarning
from langchain_core.tools import BaseTool

with warnings.catch_warnings():
    warnings.simplefilter("ignore", LangChainBetaWarning)
    from langchain.mcp import MCPAdapter

logger = logging.getLogger(__name__)


class MCPToolProvider:
    """Caches the LangChain tools adapted from the remote MCP server; re-discovers on demand."""

    def __init__(self, url: str, token: str | None, timeout_s: float = 90.0) -> None:
        self._url = url
        self._token = token
        self._timeout_s = timeout_s
        self._tools: list[BaseTool] | None = None
        self._lock = asyncio.Lock()

    def _client(self) -> Client:
        transport = StreamableHttpTransport(self._url, auth=self._token)
        return Client(transport, name="mcp-research-agent", timeout=self._timeout_s, init_timeout=15)

    async def get_tools(self, refresh: bool = False) -> list[BaseTool]:
        async with self._lock:
            if self._tools is None or refresh:
                adapter = MCPAdapter(self._client())
                self._tools = await adapter.list_tools(cache_mode="refresh")
                logger.info("Discovered %d MCP tools from %s", len(self._tools), self._url)
            return self._tools

    async def describe(self) -> list[dict]:
        """Tool name/description/schema as advertised by the MCP server (fresh discovery)."""
        async with self._client() as client:
            tools = await client.list_tools()
        return [
            {
                "name": t.name,
                "title": t.title,
                "description": t.description or "",
                "input_schema": t.input_schema,
                "output_schema": t.output_schema,
            }
            for t in tools
        ]
