"""A small, deterministic labeled dataset for the evaluation harness.

This is a **smoke set**, not a benchmark: a handful of tools and a few queries
whose ground-truth relevant tool is obvious. Its only jobs are (1) to make
``python -m mcp_router.eval`` runnable out of the box and (2) to give the test
suite a fixed, reproducible target. Nothing here should be read as a quality
claim about the retriever.

Determinism comes from the offline :class:`~mcp_router.embedder.HashingEmbedder`
(a bag-of-content-tokens embedder): the same tools and queries produce the same
vectors, and therefore the same metrics, on every machine and every run - no
model download, no network. The queries reuse vocabulary from the tool
descriptions on purpose, so the deterministic embedder ranks the intended tool
first.
"""

from __future__ import annotations

from typing import List

from .embedder import HashingEmbedder
from .eval import LabeledExample
from .models import ToolDef
from .retrieval import Retriever

#: The catalogue the smoke dataset is evaluated against. Six distinct tools so
#: each labeled query has an unambiguous best match.
SAMPLE_TOOLS: List[ToolDef] = [
    ToolDef(
        name="get_current_weather",
        description="Get the current temperature and conditions for a city.",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        upstream="weather",
    ),
    ToolDef(
        name="get_forecast",
        description="Get a multi-day weather forecast for a location.",
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
        upstream="weather",
    ),
    ToolDef(
        name="convert_currency",
        description="Convert a monetary amount from one currency to another.",
        input_schema={"type": "object", "properties": {"amount": {"type": "number"}}},
        upstream="finance",
    ),
    ToolDef(
        name="get_stock_price",
        description="Look up the latest share price for a stock ticker symbol.",
        input_schema={"type": "object", "properties": {"symbol": {"type": "string"}}},
        upstream="finance",
    ),
    ToolDef(
        name="read_file",
        description="Read the contents of a file from disk.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        upstream="files",
    ),
    ToolDef(
        name="send_email",
        description="Send an email message to one or more recipients.",
        input_schema={"type": "object", "properties": {"to": {"type": "string"}}},
        upstream="files",
    ),
]

#: The labeled queries. Each names the tool(s) that *should* be retrieved.
SAMPLE_DATASET: List[LabeledExample] = [
    LabeledExample(
        query="current temperature and conditions for a city",
        relevant_tool_names={"get_current_weather"},
    ),
    LabeledExample(
        query="multi-day weather forecast for a location",
        relevant_tool_names={"get_forecast"},
    ),
    LabeledExample(
        query="convert a monetary amount between currencies",
        relevant_tool_names={"convert_currency"},
    ),
    LabeledExample(
        query="latest share price for a stock ticker symbol",
        relevant_tool_names={"get_stock_price"},
    ),
    LabeledExample(
        query="read the contents of a file from disk",
        relevant_tool_names={"read_file"},
    ),
    LabeledExample(
        query="send an email message to a recipient",
        relevant_tool_names={"send_email"},
    ),
]


def build_sample_retriever(hybrid: bool = True, alpha: float = 0.5) -> Retriever:
    """Build a real :class:`Retriever` indexed on :data:`SAMPLE_TOOLS`.

    Forces the deterministic :class:`HashingEmbedder` rather than
    ``get_embedder()`` so the bundled dataset stays offline and reproducible
    regardless of ``MCP_ROUTER_EMBEDDER``. Reuses the production retriever - the
    harness measures the same code path the gateway serves, not a stand-in.
    """
    retriever = Retriever(HashingEmbedder(dim=256), hybrid=hybrid, alpha=alpha)
    retriever.index(SAMPLE_TOOLS)
    return retriever
