"""The tool registry: the gateway's single source of truth.

The registry ties the pieces together. From a :class:`GatewayConfig` it:

1. collects every tool from every upstream,
2. indexes them for semantic retrieval (embedder + vector store), and
3. builds the map from tool name -> upstream transport so calls can be routed.

Everything the HTTP layer needs (``list_tools``, ``retrieve``, ``call_tool``)
is exposed here, keeping the web server thin.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .cache import QueryCache, normalize_query
from .config import GatewayConfig, ServerConfig, load_config
from .embedder import Embedder, get_embedder
from .models import ToolDef
from .retrieval import Retriever
from .upstream import MockUpstream, StdioUpstream, Upstream


def _build_upstream(server: ServerConfig) -> Upstream:
    """Construct the transport for one configured upstream server."""
    transport = (server.transport or "mock").lower()
    if transport == "mock":
        return MockUpstream(server.name, tools=server.tools)
    if transport == "stdio":
        return StdioUpstream(
            server.name,
            command=server.command,
            args=server.args,
            env=server.env,
        )
    raise ValueError(
        f"server {server.name!r}: unknown transport {server.transport!r} "
        f"(expected 'mock' or 'stdio')"
    )


class ToolRegistry:
    """Owns the tool index and upstream routing table."""

    def __init__(self, config: GatewayConfig, embedder: Embedder | None = None) -> None:
        self.config = config
        # Held so we can rebuild the retriever from scratch on refresh() - a
        # Retriever's vector store is append-only, so re-indexing means a new one.
        self._embedder = embedder or get_embedder()
        self.retriever = Retriever(self._embedder)
        self._upstreams: Dict[str, Upstream] = {}

        # The query-result cache sits in front of retrieve(); see that method.
        cc = config.cache
        self._cache = QueryCache(
            enabled=cc.enabled,
            ttl_seconds=cc.ttl_seconds,
            max_entries=cc.max_entries,
        )

        self._discover_and_index()

    def _discover_and_index(self) -> None:
        """Build each upstream, discover its tools, and index them.

        For mock upstreams the tools come straight from the config; for stdio
        upstreams they are fetched live from the real server over the wire.
        Either way the tools arrive tagged with their owning upstream, so routing
        stays correct. Extracted from ``__init__`` so ``refresh()`` can reuse it.
        """
        all_tools: List[ToolDef] = []
        try:
            for server in self.config.servers:
                upstream = _build_upstream(server)
                self._upstreams[server.name] = upstream
                all_tools.extend(upstream.list_tools())
            self.retriever.index(all_tools)
        except Exception:
            # Don't leak subprocesses if discovery/indexing fails partway.
            self.close()
            raise

    # ---- construction helpers ------------------------------------------------

    @classmethod
    def from_config_file(cls, path, embedder: Embedder | None = None) -> "ToolRegistry":
        return cls(load_config(path), embedder=embedder)

    # ---- read side -----------------------------------------------------------

    def all_tools(self) -> List[ToolDef]:
        """Every tool across every upstream (what a naive client would load)."""
        return self.retriever.all_tools()

    def tool_count(self) -> int:
        return len(self.retriever.all_tools())

    def retrieve(self, query: str, k: int) -> List[Tuple[ToolDef, float]]:
        """Top-k tools for ``query`` as ``(tool, score)`` pairs.

        This is the cached entry point: a repeated query returns the previously
        computed tool set without touching the embedder or the vector store.

        Empty/whitespace queries bypass the cache entirely - the retriever
        already short-circuits them to ``[]`` and there is nothing worth storing.
        """
        if not self._cache.enabled or not query or not query.strip():
            return self.retriever.retrieve(query, k)

        key = normalize_query(query, k)
        cached = self._cache.get(key)
        if cached is not None:
            # Hand back a fresh list so a caller mutating the result can't corrupt
            # the cached copy. The (tool, score) tuples inside are immutable.
            return list(cached)

        result = self.retriever.retrieve(query, k)
        self._cache.set(key, list(result))
        return list(result)

    def list_tools(self, query: str | None = None, k: int = 5) -> List[ToolDef]:
        """Tools to expose to the client.

        This is the core value proposition. With a ``query`` we return only the
        top-``k`` relevant tools; without one we fall back to returning all tools
        (a plain proxy), because a client asking with no context should not be
        silently starved of tools.
        """
        if query and query.strip():
            return [tool for tool, _score in self.retrieve(query, k)]
        return self.all_tools()

    # ---- call side -----------------------------------------------------------

    def call_tool(self, name: str, arguments: Dict) -> Dict:
        """Route a ``tools/call`` to the upstream that owns ``name``."""
        tool = self.retriever.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name!r}")
        upstream = self._upstreams.get(tool.upstream)
        if upstream is None:  # pragma: no cover - config guarantees this exists
            raise KeyError(f"no upstream for tool {name!r} (upstream {tool.upstream!r})")
        return upstream.call(name, arguments)

    def stats(self) -> Dict[str, Any]:
        """Operational snapshot: tool count plus cache hit/miss stats."""
        return {"tools": self.tool_count(), "cache": self._cache.stats()}

    # ---- mutation ------------------------------------------------------------

    def refresh(self) -> None:
        """Re-discover tools from every upstream and rebuild the index.

        Call this when the tool catalogue may have changed - e.g. upstreams
        reconnect or a server adds/removes tools. Because the retriever's vector
        store is append-only, we rebuild it from a clean slate rather than trying
        to patch it in place.

        Crucially, the query cache is invalidated afterwards: a tool set cached
        against the old catalogue could now be wrong (a better-matching tool may
        have appeared, or a returned tool may be gone), so serving it would be
        stale. Clearing is the safe, simple choice.
        """
        self.close()
        self._upstreams = {}
        self.retriever = Retriever(self._embedder)
        self._discover_and_index()
        self._cache.clear()

    # ---- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Tear down every upstream (e.g. terminate stdio subprocesses)."""
        for upstream in self._upstreams.values():
            try:
                upstream.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    def __enter__(self) -> "ToolRegistry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
