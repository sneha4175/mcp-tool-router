"""The tool registry: the gateway's single source of truth.

The registry ties the pieces together. From a :class:`GatewayConfig` it:

1. collects every tool from every upstream,
2. indexes them for semantic retrieval (embedder + vector store), and
3. builds the map from tool name -> upstream transport so calls can be routed.

Everything the HTTP layer needs (``list_tools``, ``retrieve``, ``call_tool``)
is exposed here, keeping the web server thin.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .config import GatewayConfig, load_config
from .embedder import Embedder, get_embedder
from .models import ToolDef
from .retrieval import Retriever
from .upstream import MockUpstream, Upstream


class ToolRegistry:
    """Owns the tool index and upstream routing table."""

    def __init__(self, config: GatewayConfig, embedder: Embedder | None = None) -> None:
        self.config = config
        self.retriever = Retriever(embedder or get_embedder())
        self._upstreams: Dict[str, Upstream] = {}

        for server in config.servers:
            # For the MVP every transport is served by a MockUpstream. When real
            # stdio support lands, this is where a StdioUpstream would be built
            # for servers whose transport == "stdio".
            self._upstreams[server.name] = MockUpstream(server.name)

        self.retriever.index(config.all_tools())

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
        """Top-k tools for ``query`` as ``(tool, score)`` pairs."""
        return self.retriever.retrieve(query, k)

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
