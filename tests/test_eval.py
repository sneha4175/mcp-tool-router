"""Tests for the retrieval evaluation harness (v0.5).

Two halves:

* the metric functions are checked against **hand-computed** values so their
  arithmetic is pinned exactly (recall@k, precision@k, reciprocal rank), plus
  the boundary cases the task calls out - perfect retrieval and a total miss;
* :func:`evaluate` is run **end-to-end** on the bundled smoke dataset with the
  deterministic hashing embedder, so the aggregate is reproducible.
"""

from __future__ import annotations

import pytest

from mcp_router.config import parse_config
from mcp_router.embedder import HashingEmbedder
from mcp_router.eval import (
    EvalReport,
    LabeledExample,
    evaluate,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from mcp_router.eval_data import (
    SAMPLE_DATASET,
    SAMPLE_TOOLS,
    build_sample_retriever,
)
from mcp_router.registry import ToolRegistry


# --- metric arithmetic, hand-computed ---------------------------------------


def test_recall_at_k_two_relevant_one_retrieved():
    """2 relevant, exactly 1 of them in the top-k -> recall = 1/2 = 0.5."""
    relevant = {"a", "b"}
    retrieved = ["a", "x", "y"]  # only "a" is relevant, and it's within k=3
    assert recall_at_k(relevant, retrieved, k=3) == pytest.approx(0.5)


def test_precision_at_k_divides_by_fixed_k():
    """precision@k uses the fixed slot count k, not the number retrieved."""
    relevant = {"a", "b"}
    retrieved = ["a", "b", "c"]
    assert precision_at_k(relevant, retrieved, k=2) == pytest.approx(1.0)   # 2/2
    assert precision_at_k(relevant, retrieved, k=3) == pytest.approx(1.0 / 3 * 2)  # 2/3
    assert precision_at_k(relevant, retrieved, k=4) == pytest.approx(0.5)   # 2/4


def test_reciprocal_rank_first_relevant_at_rank_two():
    """First relevant hit sits at rank 2 -> RR = 1/2 = 0.5."""
    relevant = {"a"}
    retrieved = ["x", "a", "b"]
    assert reciprocal_rank(relevant, retrieved) == pytest.approx(0.5)


def test_reciprocal_rank_first_relevant_at_rank_one():
    assert reciprocal_rank({"a"}, ["a", "b", "c"]) == pytest.approx(1.0)


def test_k_slicing_excludes_relevant_beyond_k():
    """A relevant item ranked below k must not count toward recall/precision."""
    relevant = {"a"}
    retrieved = ["x", "y", "a"]  # "a" is at rank 3, outside k=2
    assert recall_at_k(relevant, retrieved, k=2) == 0.0
    assert precision_at_k(relevant, retrieved, k=2) == 0.0
    # ...but reciprocal rank scans the full ranking: 1/3.
    assert reciprocal_rank(relevant, retrieved) == pytest.approx(1.0 / 3)


# --- boundary cases ---------------------------------------------------------


def test_perfect_retrieval():
    """All relevant tools returned, one at rank 1 -> recall 1.0 and RR 1.0."""
    relevant = {"a", "b"}
    retrieved = ["a", "b", "c"]
    assert recall_at_k(relevant, retrieved, k=3) == pytest.approx(1.0)
    assert precision_at_k(relevant, retrieved, k=2) == pytest.approx(1.0)
    assert reciprocal_rank(relevant, retrieved) == pytest.approx(1.0)


def test_no_hit_case():
    """Nothing relevant retrieved -> recall 0, precision 0, RR 0."""
    relevant = {"a", "b"}
    retrieved = ["x", "y", "z"]
    assert recall_at_k(relevant, retrieved, k=3) == 0.0
    assert precision_at_k(relevant, retrieved, k=3) == 0.0
    assert reciprocal_rank(relevant, retrieved) == 0.0


def test_metric_guards():
    """Degenerate inputs are guarded, not crashes."""
    assert recall_at_k(set(), ["a"], k=3) == 0.0        # no relevant -> guard
    assert precision_at_k({"a"}, ["a"], k=0) == 0.0     # k=0 -> guard
    assert reciprocal_rank({"a"}, []) == 0.0            # empty ranking


# --- evaluate() end-to-end on the smoke dataset -----------------------------


def test_evaluate_on_sample_dataset_is_sane_and_perfect():
    """The crafted smoke dataset is fully retrievable by the hashing embedder:
    every query's one relevant tool is returned at rank 1."""
    retriever = build_sample_retriever()
    report = evaluate(retriever, SAMPLE_DATASET, k=3)

    assert isinstance(report, EvalReport)
    assert report.k == 3
    assert len(report.per_query) == len(SAMPLE_DATASET) == 6

    # Every metric is a valid fraction.
    for r in report.per_query:
        assert 0.0 <= r.recall <= 1.0
        assert 0.0 <= r.precision <= 1.0
        assert 0.0 <= r.reciprocal_rank <= 1.0

    # Each query labels exactly one relevant tool, retrieved at rank 1.
    assert report.mean_recall == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)
    # 1 relevant tool over k=3 slots -> precision 1/3 for every query.
    assert report.mean_precision == pytest.approx(1.0 / 3)


def test_evaluate_is_deterministic():
    """Same tools + same queries + hashing embedder -> identical aggregate."""
    a = evaluate(build_sample_retriever(), SAMPLE_DATASET, k=3)
    b = evaluate(build_sample_retriever(), SAMPLE_DATASET, k=3)
    assert (a.mean_recall, a.mean_precision, a.mrr) == (
        b.mean_recall,
        b.mean_precision,
        b.mrr,
    )


def test_evaluate_accepts_a_tool_registry():
    """evaluate() is duck-typed on .retrieve(query, k), so a ToolRegistry - the
    object the gateway actually serves - works without adaptation."""
    cfg = parse_config(
        {
            "servers": [
                {
                    "name": "smoke",
                    "transport": "mock",
                    "tools": [t.to_mcp() for t in SAMPLE_TOOLS],
                }
            ]
        }
    )
    registry = ToolRegistry(cfg, embedder=HashingEmbedder(dim=256))
    report = evaluate(registry, SAMPLE_DATASET, k=3)

    assert report.mean_recall == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)


def test_evaluate_empty_dataset_is_zeroed():
    report = evaluate(build_sample_retriever(), [], k=3)
    assert (report.mean_recall, report.mean_precision, report.mrr) == (0.0, 0.0, 0.0)
    assert report.per_query == []


def test_report_format_table_contains_metrics():
    report = evaluate(build_sample_retriever(), SAMPLE_DATASET, k=3)
    table = report.format_table()
    assert "recall@k" in table
    assert "MEAN" in table
    assert "k=3" in table
