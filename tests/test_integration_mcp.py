"""Real MCP transport integration tests.

Unlike the rest of the suite (which is pure in-process logic), these actually
launch MCP servers as subprocesses and speak the real protocol to them via the
official ``mcp`` SDK. They stay deterministic and offline by using a small
example MCP server bundled in ``examples/`` - no network, no npx, no model.

Two paths are exercised:

1. The gateway as a real MCP **client** (``StdioUpstream`` and a ``ToolRegistry``
   whose tools are discovered live from the example server).
2. The gateway as a real MCP **server** (``create_mcp_server``): a real
   ``Client`` connects to it in-memory and drives the full loop -
   find_tools -> call_tool -> proxied to the real upstream -> real result.

A third test connects to the official off-the-shelf filesystem server via
``npx``; it needs network + Node and is skipped unless
``MCP_ROUTER_RUN_NPX_TESTS=1`` is set, so the default suite never depends on it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from mcp_router.config import parse_config
from mcp_router.embedder import HashingEmbedder
from mcp_router.registry import ToolRegistry
from mcp_router.upstream import StdioUpstream

EXAMPLE_SERVER = Path(__file__).resolve().parent.parent / "examples" / "example_upstream_server.py"

# The example server's tools, for assertions.
EXAMPLE_TOOLS = {"echo", "add_numbers", "reverse_text", "uppercase"}


def _example_stdio_config() -> dict:
    """A config with the example server as a real stdio upstream + a mock one."""
    return {
        "servers": [
            {
                "name": "example",
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(EXAMPLE_SERVER)],
            },
            {
                "name": "weather",
                "transport": "mock",
                "tools": [
                    {
                        "name": "get_forecast",
                        "description": "Get a multi-day weather forecast for a location.",
                        "inputSchema": {"type": "object", "properties": {"location": {"type": "string"}}},
                    }
                ],
            },
        ]
    }


def test_stdio_upstream_discovers_real_tools_and_calls():
    """The gateway, as a real MCP client, lists and calls a real server's tools."""
    up = StdioUpstream("example", command=sys.executable, args=[str(EXAMPLE_SERVER)])
    try:
        tools = up.list_tools()
        names = {t.name for t in tools}
        assert EXAMPLE_TOOLS.issubset(names)
        # Every discovered tool is tagged with the owning upstream (needed for routing).
        assert all(t.upstream == "example" for t in tools)

        result = up.call("echo", {"message": "hello mcp"})
        assert result["isError"] is False
        assert result["_upstream"] == "example"
        text = " ".join(b.get("text", "") for b in result["content"])
        assert "hello mcp" in text
    finally:
        up.close()


def test_registry_indexes_real_upstream_and_routes_call():
    """End-to-end through the registry: real discovery + retrieval + routed call."""
    with ToolRegistry(parse_config(_example_stdio_config()), embedder=HashingEmbedder(dim=256)) as reg:
        # Tools from BOTH the real stdio upstream and the mock upstream are indexed.
        names = {t.name for t in reg.all_tools()}
        assert EXAMPLE_TOOLS.issubset(names)
        assert "get_forecast" in names

        # Retrieval finds the real tool for a relevant query...
        hits = reg.retrieve("reverse this text string", k=1)
        assert hits and hits[0][0].name == "reverse_text"

        # ...and a call is proxied to the real upstream, returning its real output.
        result = reg.call_tool("reverse_text", {"text": "abcedf"})
        assert result["_upstream"] == "example"
        text = " ".join(b.get("text", "") for b in result["content"])
        assert "fdecba" in text  # the example server actually reversed it


def test_router_as_real_mcp_server_full_loop():
    """The gateway *is* a real MCP server: a real Client drives find_tools+call_tool."""
    import anyio

    from mcp import Client

    from mcp_router.mcp_server import create_mcp_server

    reg = ToolRegistry(parse_config(_example_stdio_config()), embedder=HashingEmbedder(dim=256))
    server = create_mcp_server(reg, default_k=3)

    async def scenario():
        async with Client(server) as client:
            listed = await client.list_tools()
            exposed = {t.name for t in listed.tools}
            # A host loads just the two meta-tools, not every upstream tool.
            assert exposed == {"find_tools", "call_tool"}

            found = await client.call_tool("find_tools", {"query": "make text uppercase", "k": 2})
            found_text = found.content[0].text
            assert "uppercase" in found_text

            called = await client.call_tool(
                "call_tool", {"name": "uppercase", "arguments": {"text": "abc"}}
            )
            assert "ABC" in called.content[0].text

    try:
        anyio.run(scenario)
    finally:
        reg.close()


@pytest.mark.skipif(
    os.environ.get("MCP_ROUTER_RUN_NPX_TESTS") != "1",
    reason="needs network + Node/npx; set MCP_ROUTER_RUN_NPX_TESTS=1 to run",
)
def test_npx_filesystem_upstream_real():
    """Optional: connect to the official @modelcontextprotocol/server-filesystem."""
    up = StdioUpstream(
        "filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(Path.cwd())],
        connect_timeout=180,
    )
    try:
        names = {t.name for t in up.list_tools()}
        assert "read_text_file" in names or "read_file" in names
        result = up.call("list_allowed_directories", {})
        assert result["isError"] is False
    finally:
        up.close()
