"""Lexical (keyword-overlap) scoring for hybrid retrieval (v0.4).

Pure-embedding retrieval measures *meaning*, which is exactly what you want for
"schedule a meeting" -> ``create_event``. But it has a blind spot: an exact
tool-name or keyword hit can score only moderately when the surrounding words
differ, so a semantically-fuzzy distractor can edge out the tool the user
literally named. A lexical signal fixes that by rewarding shared surface tokens.

This module contributes the lexical half of the hybrid score. It is deliberately
tiny and dependency-free - no BM25 corpus statistics, no IDF table to keep in
sync with the index. For a catalogue of tens-to-hundreds of tools, a normalized
token-overlap is both sufficient and easy to reason about.

The metric is **query coverage**: of the meaningful tokens in the query, what
fraction appear in the tool's text (name + description + parameter names)?

    lexical = |query_tokens ∩ doc_tokens| / |query_tokens|

Properties that make it a good partner for cosine similarity:

* **Bounded to [0, 1]**, like the cosine score, so blending them with a single
  weight ``alpha`` is meaningful without rescaling.
* **Query-anchored.** We normalize by the query length, not the document, so a
  long, richly-described tool is not penalized for having extra words - it just
  needs to *contain* what the user asked for.
* **Stopword-aware.** It reuses the embedder's ``content_tokens`` so filler
  words ("the", "a", "for") never manufacture a match.

Trade-off, stated plainly: coverage ignores term frequency (a token counts once)
and rarity (every token is weighted equally). BM25 models both. We skip it on
purpose - the added corpus bookkeeping is not worth it at this scale, and the
hybrid blend already leans on the embedding for nuance.
"""

from __future__ import annotations

from typing import AbstractSet

from .embedder import content_tokens


def doc_tokens(text: str) -> frozenset[str]:
    """The lexical token set for a tool's text, stopwords removed.

    Returned as a ``frozenset`` because per-tool token sets are computed once at
    index time and only ever membership-tested afterwards.
    """
    return frozenset(content_tokens(text))


def lexical_score(query: str, tokens: AbstractSet[str]) -> float:
    """Fraction of the query's content tokens present in ``tokens`` (0..1).

    ``tokens`` is a pre-computed document token set (see :func:`doc_tokens`), so
    scoring a query against every tool is just set membership - cheap enough to
    run over the whole catalogue on each query.

    An empty or all-stopword query has no tokens to cover and scores 0.0, which
    mirrors the retriever's "no context -> no arbitrary pick" stance.
    """
    q = set(content_tokens(query))
    if not q:
        return 0.0
    hits = sum(1 for tok in q if tok in tokens)
    return hits / len(q)
