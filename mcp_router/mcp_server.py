"""The gateway as a *real* MCP server, over the official ``mcp`` SDK.

The FastAPI surface (``server.py``) speaks an MCP-shaped JSON-RPC with a custom
``query`` extension on ``tools/list``. That is convenient for HTTP callers, but a
standard MCP host (Claude Desktop, etc.) connects over stdio and calls a vanilla
``tools/list`` that has **no** place to put a query. So how does the gateway
deliver its "N tools -> k tools" reduction to a real host?

**Progressive disclosure.** Instead of advertising all N upstream tools up front
(which would defeat the whole purpose), this MCP server exposes just two
meta-tools:

* ``find_tools(query, k)`` - semantic search over every upstream tool. Returns
  the top-k matching tool definitions (name, description, input schema).
* ``call_tool(name, arguments)`` - proxy an actual call to whichever upstream
  owns ``name``, returning that server's real result.

A host therefore loads **two** tool definitions instead of N. When the model
needs a capability it searches for it, then calls it. The context overhead
becomes constant regardless of how many upstream servers are connected - which is
the entire thesis of the project, realised over the real protocol.

Run it as the server a host launches::

    python -m mcp_router.mcp_server        # serves on stdio

or, in code, connect to it in-memory with the SDK's ``Client`` for testing.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from mcp.server.mcpserver import MCPServer

from .registry import ToolRegistry

DEFAULT_K = int(os.environ.get("MCP_ROUTER_DEFAULT_K", "5"))


def create_mcp_server(registry: ToolRegistry, default_k: int = DEFAULT_K) -> MCPServer:
    """Build an :class:`MCPServer` that fronts ``registry`` via two meta-tools."""

    server = MCPServer(
        name="mcp-tool-router",
        version="0.4.0",
        instructions=(
            "A retrieval gateway over many MCP servers. Call find_tools(query) to "
            "discover the few tools relevant to your task, then call_tool(name, "
            "arguments) to invoke one. This keeps only a handful of tool "
            "definitions in context instead of every tool from every server."
        ),
    )

    @server.tool()
    def find_tools(query: str, k: int = default_k) -> str:
        """Search all connected MCP servers for the tools most relevant to a
        natural-language query. Returns up to k tool definitions (name,
        description, input schema) as JSON. Call this first to discover which
        tool to use, then invoke it with call_tool."""
        hits = registry.retrieve(query, k=k)
        payload = {
            "query": query,
            "total_tools_available": registry.tool_count(),
            "returned": len(hits),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                    "upstream": tool.upstream,
                    "score": round(score, 4),
                }
                for tool, score in hits
            ],
        }
        return json.dumps(payload, indent=2)

    @server.tool()
    def call_tool(name: str, arguments: Dict[str, Any] | None = None) -> str:
        """Invoke a tool by name (as returned by find_tools) on whichever
        upstream server owns it, forwarding the given arguments, and return that
        server's result."""
        try:
            result = registry.call_tool(name, arguments or {})
        except KeyError as exc:
            return f"error: {exc}"

        # Surface the upstream's text content to the host; fall back to the raw
        # structured result for non-text blocks.
        texts = [
            block.get("text", "")
            for block in result.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if texts:
            body = "\n".join(t for t in texts if t)
        else:
            body = json.dumps(result.get("content", []), indent=2)
        if result.get("isError"):
            return f"[upstream error] {body}"
        return body

    return server


def build_from_env() -> ToolRegistry:
    """Construct the registry from ``MCP_ROUTER_CONFIG`` (used by ``__main__``)."""
    config_path = os.environ.get("MCP_ROUTER_CONFIG", "config.example.yaml")
    return ToolRegistry.from_config_file(config_path)


def main() -> None:
    """Serve the gateway as an MCP server over stdio (what a host launches)."""
    registry = build_from_env()
    server = create_mcp_server(registry)
    try:
        server.run("stdio")
    finally:
        registry.close()


if __name__ == "__main__":
    main()
