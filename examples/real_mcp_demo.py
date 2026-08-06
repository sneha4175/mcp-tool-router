"""End-to-end demo against REAL MCP servers.

Unlike ``demo.py`` (which uses inline mock tools), this connects the gateway to
genuine MCP servers over stdio using the official SDK, then shows the whole win:

    N real tools, discovered live from real servers  ->  k exposed per query
    ...and a real tools/call proxied back to the upstream that owns the tool.

Upstreams used:

* ``examples/example_upstream_server.py`` - a small real MCP server in this repo
  (always available; no network, no downloads).
* The official ``@modelcontextprotocol/server-filesystem`` via ``npx`` - a real,
  off-the-shelf server. Used automatically if ``npx`` is on PATH; skipped with a
  note otherwise, so the demo still runs fully offline.

Embedder: whatever ``MCP_ROUTER_EMBEDDER`` selects. The v0.2 default is the real
``sentence-transformers`` model (downloads ~80MB once). For a fast, offline run::

    MCP_ROUTER_EMBEDDER=hashing python examples/real_mcp_demo.py

Run it::

    python examples/real_mcp_demo.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from mcp_router.config import parse_config
from mcp_router.registry import ToolRegistry

EXAMPLE_SERVER = Path(__file__).resolve().parent / "example_upstream_server.py"
TOP_K = 3

QUERIES = [
    "reverse the letters in a word",
    "add two numbers together",
    "read the contents of a text file",
    "list the files inside a directory",
]


def build_config() -> dict:
    servers = [
        {
            "name": "example",
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(EXAMPLE_SERVER)],
        }
    ]
    if shutil.which("npx"):
        servers.append(
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(Path.cwd())],
            }
        )
    else:
        print("(!) npx not found - running with the example server only.\n")
    return {"servers": servers}


def main() -> None:
    embedder = os.environ.get("MCP_ROUTER_EMBEDDER", "sentence-transformers")
    print("=" * 70)
    print("MCP Tool-Retrieval Gateway - REAL upstream demo (v0.2)")
    print("=" * 70)
    print(f"Embedder: {embedder}")
    print("Connecting to real MCP servers over stdio...\n")

    with ToolRegistry(parse_config(build_config())) as reg:
        total = reg.tool_count()
        print(f"Upstream MCP servers : {len(reg.config.servers)}  (real, live over stdio)")
        print(f"Real tools discovered: {total}  (a naive client would load all of these)")
        print(f"Exposed per query    : {TOP_K}  (top-k relevant)")
        reduction = 100 * (total - TOP_K) / total if total else 0
        print(f"Reduction            : {total} -> {TOP_K}  (~{reduction:.0f}% fewer)\n")

        for query in QUERIES:
            hits = reg.retrieve(query, k=TOP_K)
            if not hits:
                continue
            print(f"query: {query!r}")
            for rank, (tool, score) in enumerate(hits, start=1):
                print(f"   {rank}. {tool.name:<22} [{tool.upstream:<10}] score={score:.3f}")
            print()

        # Prove the call path: proxy a real call to the upstream that owns the tool.
        print("-" * 70)
        result = reg.call_tool("reverse_text", {"text": "gateway"})
        text = " ".join(b.get("text", "") for b in result["content"])
        print(f"tools/call reverse_text('gateway') -> [{result['_upstream']}] {text}")


if __name__ == "__main__":
    main()
