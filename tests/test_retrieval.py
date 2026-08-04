"""Tests for semantic retrieval over tool definitions."""

from __future__ import annotations

import pytest

from mcp_router.registry import ToolRegistry


@pytest.mark.parametrize(
    "query,expected_tool",
    [
        ("what is the weather forecast tomorrow", "get_forecast"),
        ("convert dollars to euros", "convert_currency"),
        ("read the contents of a file on disk", "read_file"),
        ("send an email to my manager", "send_email"),
        ("current share price of a stock", "get_stock_price"),
    ],
)
def test_retrieve_returns_correct_top_tool(registry: ToolRegistry, query, expected_tool):
    hits = registry.retrieve(query, k=1)
    assert hits, "expected at least one hit"
    assert hits[0][0].name == expected_tool


def test_retrieve_respects_k(registry: ToolRegistry):
    hits = registry.retrieve("read a file from disk", k=3)
    assert len(hits) == 3


def test_retrieve_scores_are_descending(registry: ToolRegistry):
    hits = registry.retrieve("send an email message", k=4)
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing(registry: ToolRegistry):
    assert registry.retrieve("   ", k=5) == []


def test_duplicate_tool_names_are_rejected():
    from mcp_router.config import parse_config
    from mcp_router.embedder import HashingEmbedder

    dupe = {
        "servers": [
            {"name": "a", "tools": [{"name": "same", "description": "one"}]},
            {"name": "b", "tools": [{"name": "same", "description": "two"}]},
        ]
    }
    with pytest.raises(ValueError):
        ToolRegistry(parse_config(dupe), embedder=HashingEmbedder(dim=64))
