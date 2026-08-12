"""Tests for hybrid retrieval (v0.4): semantic + lexical blend.

Everything here runs on the deterministic HashingEmbedder, so the exact scores
are reproducible on every machine. The key fixture is a *crafted* two-tool
catalogue where a distractor wins on pure semantic similarity but the tool the
user literally named wins once lexical overlap is blended in - the whole reason
hybrid retrieval exists.

Hand-computed with the hashing embedder (bag-of-content-tokens, cosine of
L2-normalized vectors), for the query ``"delete_user account"`` whose content
tokens are {delete, user, account}:

* ``delete_user`` / "Remove an account permanently"
  -> tokens {delete, user, remove, account, permanently}
  -> cosine 3/sqrt(15) = 0.7746 ; lexical coverage 3/3 = 1.0
* ``user_account`` / ""
  -> tokens {user, account}
  -> cosine 2/sqrt(6) = 0.8165 ; lexical coverage 2/3 = 0.6667

So pure semantic ranks the distractor ``user_account`` first, while an even
blend (alpha=0.5) puts the exact-name match ``delete_user`` on top.
"""

from __future__ import annotations

import pytest

from mcp_router.config import RetrievalConfig, parse_config
from mcp_router.embedder import HashingEmbedder
from mcp_router.lexical import doc_tokens, lexical_score
from mcp_router.registry import ToolRegistry
from mcp_router.retrieval import Retriever

# A deterministic catalogue engineered so semantic and hybrid disagree.
CRAFTED_CONFIG = {
    "servers": [
        {
            "name": "accounts",
            "transport": "mock",
            "tools": [
                {
                    "name": "delete_user",
                    "description": "Remove an account permanently",
                },
                {
                    "name": "user_account",
                    "description": "",
                },
            ],
        }
    ]
}

QUERY = "delete_user account"  # contains the exact tool name "delete_user"


def _registry(hybrid: bool, alpha: float = 0.5) -> ToolRegistry:
    cfg = parse_config({**CRAFTED_CONFIG, "retrieval": {"hybrid": hybrid, "alpha": alpha}})
    # A fresh embedder per registry - the hashing embedder is stateless and
    # deterministic, so this stays fully reproducible.
    return ToolRegistry(cfg, embedder=HashingEmbedder(dim=256))


# --- the core behaviour -----------------------------------------------------


def test_exact_name_match_wins_under_hybrid_despite_lower_semantic():
    """The tool the user named tops the ranking under hybrid, even though a
    distractor scores higher on pure semantic similarity."""
    pure = _registry(hybrid=False).retrieve(QUERY, k=2)
    hybrid = _registry(hybrid=True, alpha=0.5).retrieve(QUERY, k=2)

    # Pure semantic is fooled: the distractor outranks the named tool.
    assert pure[0][0].name == "user_account"
    assert pure[0][1] > pure[1][1]

    # Hybrid rescues the exact-name match to the top.
    assert hybrid[0][0].name == "delete_user"
    assert hybrid[0][1] > hybrid[1][1]


def test_hybrid_and_pure_semantic_disagree_on_ranking():
    """The crafted case produces a genuinely different ranking, not just
    different scores."""
    pure = [t.name for t, _ in _registry(hybrid=False).retrieve(QUERY, k=2)]
    hybrid = [t.name for t, _ in _registry(hybrid=True, alpha=0.5).retrieve(QUERY, k=2)]

    assert pure == ["user_account", "delete_user"]
    assert hybrid == ["delete_user", "user_account"]
    assert pure != hybrid


# --- alpha boundaries -------------------------------------------------------


def test_alpha_one_reduces_to_pure_semantic():
    """alpha=1.0 is exactly the pure-semantic path: same order, same scores."""
    pure = _registry(hybrid=False).retrieve(QUERY, k=2)
    at_one = _registry(hybrid=True, alpha=1.0).retrieve(QUERY, k=2)

    assert [t.name for t, _ in at_one] == [t.name for t, _ in pure]
    for (_, a), (_, b) in zip(at_one, pure):
        assert a == pytest.approx(b)


def test_alpha_zero_reduces_to_pure_lexical():
    """alpha=0.0 is pure lexical: scores equal the token-overlap coverage and
    the exact-name tool (full coverage) wins outright."""
    hits = _registry(hybrid=True, alpha=0.0).retrieve(QUERY, k=2)
    by_name = {t.name: score for t, score in hits}

    assert hits[0][0].name == "delete_user"
    assert by_name["delete_user"] == pytest.approx(1.0)       # 3/3 tokens covered
    assert by_name["user_account"] == pytest.approx(2.0 / 3)  # 2/3 tokens covered


# --- the lexical primitive --------------------------------------------------


def test_lexical_score_is_query_coverage():
    tokens = doc_tokens("Remove an account permanently")  # {remove, account, permanently}
    # Only "account" of the three query tokens is present -> 1/3 coverage.
    assert lexical_score("delete user account", tokens) == pytest.approx(1.0 / 3)
    assert lexical_score("account", tokens) == pytest.approx(1.0)
    assert lexical_score("remove account permanently", tokens) == pytest.approx(1.0)


def test_lexical_score_empty_or_all_stopwords_is_zero():
    tokens = doc_tokens("read a file from disk")
    assert lexical_score("", tokens) == 0.0
    assert lexical_score("the and for", tokens) == 0.0  # all stopwords


# --- config parsing + defaults ----------------------------------------------


def test_retrieval_config_defaults_hybrid_on():
    cfg = parse_config(CRAFTED_CONFIG)  # no 'retrieval' block
    assert cfg.retrieval == RetrievalConfig(hybrid=True, alpha=0.5)


def test_retrieval_config_parses_overrides():
    cfg = parse_config({**CRAFTED_CONFIG, "retrieval": {"hybrid": False, "alpha": 0.25}})
    assert cfg.retrieval.hybrid is False
    assert cfg.retrieval.alpha == pytest.approx(0.25)


def test_retrieval_config_rejects_out_of_range_alpha():
    with pytest.raises(ValueError):
        parse_config({**CRAFTED_CONFIG, "retrieval": {"alpha": 1.5}})


def test_retriever_rejects_out_of_range_alpha():
    with pytest.raises(ValueError):
        Retriever(HashingEmbedder(dim=32), hybrid=True, alpha=-0.1)


def test_disabled_hybrid_is_semantic_passthrough():
    """With hybrid off the registry's retrieve matches the raw cosine store
    search - the v0.3 behaviour is preserved untouched."""
    reg = _registry(hybrid=False)
    assert reg.retriever.hybrid is False

    got = reg.retrieve(QUERY, k=2)
    # Compare against the vector store directly: same order, same scores.
    q_vec = reg.retriever.embedder.embed_one(QUERY)
    raw = reg.retriever.store.search(q_vec, 2)
    assert [t.name for t, _ in got] == [name for name, _ in raw]
    for (_, a), (_, b) in zip(got, raw):
        assert a == pytest.approx(b)


def test_stats_reports_retrieval_mode():
    stats = _registry(hybrid=True, alpha=0.5).stats()
    assert stats["retrieval"] == {"hybrid": True, "alpha": 0.5}
