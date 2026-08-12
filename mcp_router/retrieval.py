"""Semantic retrieval over tool definitions.

This is the heart of the project. Given a natural-language query it returns the
tools most likely to be relevant, so the gateway can expose only those instead
of every tool from every upstream. It wires together an :class:`Embedder` and a
:class:`VectorStore`:

    index:   ToolDef -> embedding_text -> embed() -> VectorStore.add()
    query:   text    -> embed()        -> VectorStore.search() -> ToolDefs

Both halves use the same embedder, which is the only way the numbers are
comparable.

**Hybrid retrieval (v0.4).** Pure cosine similarity captures meaning but can miss
an exact tool-name or keyword hit. When ``hybrid`` is enabled the retriever
blends the semantic (cosine) score with a lexical token-overlap score (see
:mod:`mcp_router.lexical`)::

    final = alpha * semantic + (1 - alpha) * lexical

so a tool the user literally named surfaces even when its embedding similarity is
only moderate. ``alpha`` slides between the two signals: ``1.0`` is pure
semantic (the v0.3 behaviour), ``0.0`` is pure lexical. Because a high-lexical
tool may sit outside the semantic top-k, the hybrid path scores the *whole*
catalogue before taking the top-k - affordable since a gateway fronts only
tens-to-hundreds of tools.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .embedder import Embedder
from .lexical import doc_tokens, lexical_score
from .models import ToolDef
from .vectorstore import VectorStore

#: Default blend weight when hybrid is on: an even split between the semantic and
#: lexical signals. Also the config default (see :class:`RetrievalConfig`).
DEFAULT_ALPHA = 0.5


class Retriever:
    """Indexes tools and retrieves the top-k relevant ones for a query.

    With ``hybrid=False`` this is the v0.3 pure-cosine retriever. With
    ``hybrid=True`` (the default from config) it blends cosine with a lexical
    token-overlap score weighted by ``alpha`` - see the module docstring.
    """

    def __init__(
        self,
        embedder: Embedder,
        hybrid: bool = False,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
        self.embedder = embedder
        self.hybrid = hybrid
        self.alpha = float(alpha)
        self.store = VectorStore(dim=embedder.dim)
        self._by_name: Dict[str, ToolDef] = {}
        # Per-tool lexical token sets, built once at index time so hybrid scoring
        # is pure set-membership at query time (no re-tokenizing tool text).
        self._doc_tokens: Dict[str, frozenset[str]] = {}

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
            self._doc_tokens[tool.name] = doc_tokens(tool.embedding_text())
            self.store.add(tool.name, vector)

    def get(self, name: str) -> ToolDef | None:
        return self._by_name.get(name)

    def all_tools(self) -> List[ToolDef]:
        return list(self._by_name.values())

    def retrieve(self, query: str, k: int) -> List[Tuple[ToolDef, float]]:
        """Return up to ``k`` ``(tool, score)`` pairs, best match first.

        An empty or whitespace-only query yields no results rather than
        arbitrary ones - "no context" should not silently pick tools.

        Routes to the hybrid blend when ``self.hybrid`` is set, otherwise the
        pure-cosine path. Both return scores in the same ``[0, 1]`` range.
        """
        if not query or not query.strip():
            return []
        if self.hybrid:
            return self._retrieve_hybrid(query, k)
        q_vec = self.embedder.embed_one(query)
        hits = self.store.search(q_vec, k)
        return [(self._by_name[name], score) for name, score in hits]

    def _retrieve_hybrid(self, query: str, k: int) -> List[Tuple[ToolDef, float]]:
        """Blend semantic cosine with lexical overlap, then take the top-k.

        Scores the *entire* catalogue rather than reranking the semantic top-k:
        a tool with a strong keyword match but only moderate embedding
        similarity could otherwise never enter the candidate set. ``search`` with
        ``k = len(store)`` returns every ``(name, cosine)`` pair, which we blend
        with the lexical score and re-sort. ``store.search`` already yields
        results in descending-cosine order, so equal blended scores keep a stable,
        semantic-first tiebreak.
        """
        if k <= 0 or not self._by_name:
            return []
        q_vec = self.embedder.embed_one(query)
        semantic = self.store.search(q_vec, len(self._by_name))

        blended: List[Tuple[ToolDef, float]] = []
        for name, sem_score in semantic:
            lex_score = lexical_score(query, self._doc_tokens[name])
            score = self.alpha * sem_score + (1.0 - self.alpha) * lex_score
            blended.append((self._by_name[name], score))

        blended.sort(key=lambda pair: pair[1], reverse=True)
        return blended[:k]
