"""A tiny, real MCP server used as an upstream in demos and tests.

This is a genuine MCP server built on the official ``mcp`` SDK (``MCPServer``).
It speaks the real protocol over stdio, so the gateway can connect to it as a
real MCP *client* without any network access or external downloads - which is
exactly what makes the integration tests deterministic and offline.

Run it directly to serve on stdio (this is what a config's ``stdio`` transport
launches as a subprocess)::

    python examples/example_upstream_server.py

The tool descriptions are deliberately varied so semantic retrieval has
something meaningful to rank.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

server = MCPServer(name="example-upstream", version="0.2.0")


@server.tool()
def echo(message: str) -> str:
    """Echo a message straight back to the caller."""
    return f"echo: {message}"


@server.tool()
def add_numbers(a: float, b: float) -> str:
    """Add two numbers together and return their arithmetic sum."""
    return f"{a} + {b} = {a + b}"


@server.tool()
def reverse_text(text: str) -> str:
    """Reverse the characters of a piece of text."""
    return text[::-1]


@server.tool()
def uppercase(text: str) -> str:
    """Convert a piece of text to upper case letters."""
    return text.upper()


if __name__ == "__main__":
    # Blocking call; serves the MCP protocol over stdin/stdout.
    server.run("stdio")
