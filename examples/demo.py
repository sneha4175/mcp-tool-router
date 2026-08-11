"""End-to-end demo: how many tools does the gateway hide?

Loads ``config.example.yaml``, then for a handful of realistic queries shows how
many tools a naive MCP client would load (all of them) versus how many this
gateway exposes (top-k). Run it offline:

    python examples/demo.py

No network, no model download - it uses the deterministic hashing embedder.
"""

from __future__ import annotations

from pathlib import Path

from mcp_router.registry import ToolRegistry

CONFIG = Path(__file__).resolve().parent.parent / "config.example.yaml"
TOP_K = 3

QUERIES = [
    "what's the weather forecast for tomorrow",
    "convert 100 US dollars into euros",
    "read the contents of a config file on disk",
    "schedule a meeting with the team next week",
    "send an email to my manager about the report",
]


def main() -> None:
    registry = ToolRegistry.from_config_file(CONFIG)
    total = registry.tool_count()

    print("=" * 68)
    print("MCP Tool-Retrieval Gateway - context reduction demo")
    print("=" * 68)
    print(f"Upstream servers : {len(registry.config.servers)}")
    print(f"Total tools       : {total}  (what a naive client loads every request)")
    print(f"Exposed per query : {TOP_K}  (top-k relevant tools)")
    reduction = 100 * (total - TOP_K) / total
    print(f"Tool-definition reduction: {total} -> {TOP_K}  (~{reduction:.0f}% fewer)")
    print()

    for query in QUERIES:
        hits = registry.retrieve(query, k=TOP_K)
        print(f"query: {query!r}")
        for rank, (tool, score) in enumerate(hits, start=1):
            print(f"   {rank}. {tool.name:<22} [{tool.upstream:<8}] score={score:.3f}")
        print()

    # Show the routing side too: a call goes to the upstream that owns the tool.
    result = registry.call_tool("convert_currency", {"amount": 100, "from_currency": "USD"})
    print("tools/call convert_currency ->", result["content"][0]["text"])

    # v0.3: the query cache. Re-issuing the same queries is served from cache -
    # no re-embedding, no vector search. The stats show the resulting hits.
    for query in QUERIES:
        registry.retrieve(query, k=TOP_K)  # each is a repeat -> a cache hit
    cache = registry.stats()["cache"]
    print()
    print(f"query cache: {cache['hits']} hits / {cache['misses']} misses "
          f"(hit rate {cache['hit_rate']:.0%}, {cache['size']} entries cached)")


if __name__ == "__main__":
    main()
