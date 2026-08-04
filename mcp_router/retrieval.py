"""Semantic retrieval over tool definitions.

This is the heart of the project. Given a natural-language query it returns the
tools most likely to be relevant, so the gateway can expose only those instead
of every tool from every upstream. It wires together an :class:`Embedder` and a
:class:`VectorStore`:

    index:   ToolDef -> embedding_text -> embed() -> VectorStore.add()
    query:   text    -> embed()        -> VectorStore.search() -> ToolDefs

Both halves use the same embedder, which is the only way the numbers are
comparable.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .embedder import Embedder
from .models import ToolDef
from .vectorstore import VectorStore


class Retriever:
    """Indexes tools and retrieves the top-k relevant ones for a query."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.store = VectorStore(dim=embedder.dim)
        self._by_name: Dict[str, ToolDef] = {}

    def index(self, tools: Sequence[ToolDef]) -> None:
        """Embed and store a batch of tools.

        Batched embedding is intentional: real embedders are dramatically faster
        when handed all texts at once versus one call per tool.
        """
        if not tools:
            return
        texts = [t.embedding_text() for t in tools]
        vectors = self.embedder.embed(texts)
        for tool, vector in zip(tools, vectors):
            if tool.name in self._by_name:
                raise ValueError(f"duplicate tool name across upstreams: {tool.name!r}")
            self._by_name[tool.name] = tool
            self.store.add(tool.name, vector)

    def get(self, name: str) -> ToolDef | None:
        return self._by_name.get(name)

    def all_tools(self) -> List[ToolDef]:
        return list(self._by_name.values())

    def retrieve(self, query: str, k: int) -> List[Tuple[ToolDef, float]]:
        """Return up to ``k`` ``(tool, score)`` pairs, best match first.

        An empty or whitespace-only query yields no results rather than
        arbitrary ones - "no context" should not silently pick tools.
        """
        if not query or not query.strip():
            return []
        q_vec = self.embedder.embed_one(query)
        hits = self.store.search(q_vec, k)
        return [(self._by_name[name], score) for name, score in hits]
