"""Retrieval evaluation harness (v0.5).

Retrieval quality is the whole point of this gateway: when a query is asked, do
the *right* tools come back? Up to v0.4 that was asserted only by hand-crafted
unit tests. This module makes it measurable on a **labeled dataset** using the
standard information-retrieval metrics, so a change to the embedder, the hybrid
blend, or ``alpha`` can be judged by a number rather than a hunch.

The three metrics, over the retriever's top-``k`` results for a query whose
ground-truth relevant tools are ``R``:

* **recall@k**    = ``|R ∩ retrieved_k| / |R|`` - of the tools that *should*
  come back, what fraction did, within the top-k.
* **precision@k** = ``|R ∩ retrieved_k| / k`` - of the k slots returned, what
  fraction were relevant. (Divided by a fixed ``k``, the textbook definition.)
* **reciprocal rank (RR)** = ``1 / rank`` of the first relevant hit, or ``0`` if
  none appears; **MRR** is the mean of RR across all queries - it rewards
  putting a relevant tool *near the top*.

The harness never reimplements retrieval: :func:`evaluate` calls the real
``retriever.retrieve(query, k)`` (a :class:`~mcp_router.retrieval.Retriever` or a
:class:`~mcp_router.registry.ToolRegistry` - anything with that method) and
scores whatever names come back. The bundled dataset (see
:mod:`mcp_router.eval_data`) is a small, deterministic **smoke set** meant to
prove the harness runs and to guard against regressions - not a benchmark or a
quality claim.

Run it::

    python -m mcp_router.eval          # k=3 by default
    python -m mcp_router.eval --k 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Protocol, Sequence, Set, Tuple

from .models import ToolDef


# --- the labeled example ----------------------------------------------------


@dataclass(frozen=True)
class LabeledExample:
    """One evaluation query and the tools that *should* be retrieved for it.

    ``relevant_tool_names`` is the ground truth: the set of tool names a correct
    retriever ought to surface. It is a set because relevance is unordered and
    membership is all the metrics need.
    """

    query: str
    relevant_tool_names: Set[str]


class _SupportsRetrieve(Protocol):
    """Structural type for anything the harness can evaluate.

    Both :class:`~mcp_router.retrieval.Retriever` and
    :class:`~mcp_router.registry.ToolRegistry` satisfy this, so the harness works
    against either without importing or depending on a concrete class.
    """

    def retrieve(self, query: str, k: int) -> List[Tuple[ToolDef, float]]:
        ...


# --- the metrics (pure functions over names) --------------------------------
#
# Each takes the ground-truth ``relevant`` names and the *ranked* list of
# ``retrieved`` names (best first). They are deliberately free of any retriever
# object so they can be unit-tested against hand-computed values.


def recall_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    """``|relevant ∩ retrieved[:k]| / |relevant|``.

    An empty ``relevant`` set has nothing to recall; we return ``0.0`` as a
    degenerate guard rather than dividing by zero (real datasets always label at
    least one relevant tool per query).
    """
    if not relevant:
        return 0.0
    top_k = set(list(retrieved)[:k])
    return len(relevant & top_k) / len(relevant)


def precision_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    """``|relevant ∩ retrieved[:k]| / k``.

    Divided by the fixed slot count ``k`` (the textbook definition), so
    retrieving fewer than ``k`` items - or padding with irrelevant ones - is
    penalised. ``k <= 0`` returns ``0.0``.
    """
    if k <= 0:
        return 0.0
    top_k = set(list(retrieved)[:k])
    return len(relevant & top_k) / k


def reciprocal_rank(relevant: Set[str], retrieved: Sequence[str]) -> float:
    """``1 / rank`` of the first relevant hit (rank counts from 1), else ``0.0``."""
    for rank, name in enumerate(retrieved, start=1):
        if name in relevant:
            return 1.0 / rank
    return 0.0


# --- the aggregate report ---------------------------------------------------


@dataclass(frozen=True)
class QueryResult:
    """Per-query metrics plus the ranked names that produced them."""

    query: str
    relevant: Set[str]
    retrieved: List[str]
    recall: float
    precision: float
    reciprocal_rank: float


@dataclass(frozen=True)
class EvalReport:
    """Aggregate metrics over a dataset, with the per-query breakdown kept."""

    k: int
    mean_recall: float
    mean_precision: float
    mrr: float
    per_query: List[QueryResult] = field(default_factory=list)

    def format_table(self) -> str:
        """Render the report as a plain-text table (used by the CLI)."""
        lines = [
            f"Retrieval evaluation  (k={self.k}, queries={len(self.per_query)})",
            "",
            f"{'query':<34}{'recall@k':>10}{'prec@k':>10}{'RR':>8}",
            "-" * 62,
        ]
        for r in self.per_query:
            q = r.query if len(r.query) <= 33 else r.query[:30] + "..."
            lines.append(
                f"{q:<34}{r.recall:>10.3f}{r.precision:>10.3f}{r.reciprocal_rank:>8.3f}"
            )
        lines += [
            "-" * 62,
            f"{'MEAN':<34}{self.mean_recall:>10.3f}{self.mean_precision:>10.3f}"
            f"{self.mrr:>8.3f}",
        ]
        return "\n".join(lines)


def evaluate(
    retriever: _SupportsRetrieve,
    dataset: Sequence[LabeledExample],
    k: int,
) -> EvalReport:
    """Run every query through ``retriever`` and score the results.

    For each example the real ``retriever.retrieve(query, k)`` is called and the
    returned tools' names (in rank order) are fed to the metric functions. The
    aggregate means are computed over the dataset; an empty dataset yields zeros.
    """
    per_query: List[QueryResult] = []
    for example in dataset:
        hits = retriever.retrieve(example.query, k)
        retrieved = [tool.name for tool, _score in hits]
        per_query.append(
            QueryResult(
                query=example.query,
                relevant=set(example.relevant_tool_names),
                retrieved=retrieved,
                recall=recall_at_k(example.relevant_tool_names, retrieved, k),
                precision=precision_at_k(example.relevant_tool_names, retrieved, k),
                reciprocal_rank=reciprocal_rank(example.relevant_tool_names, retrieved),
            )
        )

    n = len(per_query)
    if n == 0:
        return EvalReport(k=k, mean_recall=0.0, mean_precision=0.0, mrr=0.0)

    return EvalReport(
        k=k,
        mean_recall=sum(r.recall for r in per_query) / n,
        mean_precision=sum(r.precision for r in per_query) / n,
        mrr=sum(r.reciprocal_rank for r in per_query) / n,
        per_query=per_query,
    )


# --- CLI --------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate the bundled smoke dataset and print the metrics table.

    Uses the deterministic hashing embedder (see
    :func:`mcp_router.eval_data.build_sample_retriever`) so the numbers are the
    same on every machine and never touch the network.
    """
    parser = argparse.ArgumentParser(
        prog="python -m mcp_router.eval",
        description="Measure retrieval quality (recall@k, precision@k, MRR) on a "
        "small bundled smoke dataset.",
    )
    parser.add_argument(
        "--k", type=int, default=3, help="top-k results to score (default: 3)"
    )
    args = parser.parse_args(argv)

    # Imported here (not at module top) so importing the metrics never drags in
    # the fixture, and to avoid an import cycle with eval_data.
    from .eval_data import SAMPLE_DATASET, build_sample_retriever

    retriever = build_sample_retriever()
    report = evaluate(retriever, SAMPLE_DATASET, k=args.k)
    print(report.format_table())
    print(
        "\nNote: this is a small, deterministic smoke dataset (hashing embedder) "
        "meant to exercise the harness - not a benchmark."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    raise SystemExit(main())
