"""Pluggable text embedders.

The gateway needs to turn tool descriptions and user queries into vectors so it
can measure semantic similarity. Embedding is deliberately abstracted behind a
small interface so the *retrieval* logic never cares which model produced the
vectors:

* ``HashingEmbedder`` - a dependency-free, deterministic embedder. It needs no
  network, no model download, and always returns the same vector for the same
  text. That makes it perfect for offline development and for tests that must be
  reproducible.
* A real embedder (e.g. ``sentence-transformers`` or a hosted API) can be
  dropped in by implementing :class:`Embedder` and pointing the
  ``MCP_ROUTER_EMBEDDER`` environment variable at it. See ``get_embedder``.

The hashing embedder uses the classic "hashing trick": every token is hashed
into one of ``dim`` buckets and its weight accumulated there. Two texts that
share vocabulary end up with overlapping non-zero buckets, so their cosine
similarity is high. It captures lexical overlap rather than deep semantics, but
that is enough to demonstrate and test the routing behaviour without pulling in
a heavy model.
"""

from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

# Split on any run of non-alphanumeric characters, then lowercase. Simple, but
# good enough for tool names/descriptions which are mostly plain English.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small English stopword list. For a lexical-overlap embedder, function words
# like "the"/"for"/"a" are pure noise: they appear in almost every tool
# description and would let an unrelated tool match a query just because both
# contain "for a". Dropping them sharpens retrieval markedly. This is content
# filtering, not tokenization, so it lives in the embedder - `tokenize` stays a
# faithful tokenizer.
_STOPWORDS = frozenset(
    """
    a an the of to for in on at by with from into onto and or but is are be
    was were been being do does did done have has had this that these those
    it its as your you my our their his her what which who whom whose when
    where why how all any some no not can could will would should i we they
    me us them then than so if else about over under out up down
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase a string and return its alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> List[str]:
    """Tokens with English stopwords removed (used for embedding)."""
    return [tok for tok in tokenize(text) if tok not in _STOPWORDS]


class Embedder(ABC):
    """Abstract base for anything that turns text into fixed-length vectors."""

    #: Dimensionality of the vectors this embedder produces.
    dim: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of strings.

        Returns a ``(len(texts), dim)`` float32 array. Batching is part of the
        contract because real embedders are far more efficient per-call when
        given many strings at once.
        """
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper to embed a single string into a 1-D vector."""
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Deterministic, offline embedder using the hashing trick.

    No parameters, no training, no downloads: identical input always yields an
    identical vector, which is exactly what reproducible tests need.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _bucket(self, token: str) -> int:
        # A stable hash across processes. Python's built-in hash() is salted per
        # run (PYTHONHASHSEED), which would make embeddings non-deterministic,
        # so we use md5 of the token bytes instead.
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in content_tokens(text):
                vectors[row, self._bucket(token)] += 1.0
        return vectors


def get_embedder() -> Embedder:
    """Return the embedder selected by environment configuration.

    ``MCP_ROUTER_EMBEDDER`` picks the implementation:

    * ``hashing`` (default) - the offline :class:`HashingEmbedder`.
    * ``sentence-transformers`` - a real semantic embedder. Loaded lazily so the
      dependency is only required when actually requested. The model name is
      taken from ``MCP_ROUTER_ST_MODEL`` (default ``all-MiniLM-L6-v2``).

    Keeping selection here means the rest of the codebase depends only on the
    :class:`Embedder` interface, never on a concrete model.
    """
    kind = os.environ.get("MCP_ROUTER_EMBEDDER", "hashing").lower()

    if kind == "hashing":
        dim = int(os.environ.get("MCP_ROUTER_HASH_DIM", "256"))
        return HashingEmbedder(dim=dim)

    if kind in ("sentence-transformers", "st"):
        return _SentenceTransformerEmbedder(
            model_name=os.environ.get("MCP_ROUTER_ST_MODEL", "all-MiniLM-L6-v2")
        )

    raise ValueError(f"Unknown MCP_ROUTER_EMBEDDER value: {kind!r}")


class _SentenceTransformerEmbedder(Embedder):
    """Thin adapter over sentence-transformers (an optional real embedder).

    This is intentionally not imported at module load time: teams running the
    offline default should not need the (large) ``sentence-transformers`` and
    ``torch`` wheels installed. The import happens in ``__init__`` so a missing
    dependency fails loudly and early with an actionable message.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only with the extra installed
            raise ImportError(
                "MCP_ROUTER_EMBEDDER=sentence-transformers requires the "
                "'sentence-transformers' package. Install it with "
                "`pip install sentence-transformers`."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - needs the model
        return np.asarray(
            self._model.encode(list(texts), convert_to_numpy=True),
            dtype=np.float32,
        )
