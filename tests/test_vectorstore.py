"""Unit tests for the cosine vector store."""

from __future__ import annotations

import numpy as np
import pytest

from mcp_router.vectorstore import VectorStore, l2_normalize


def test_l2_normalize_unit_length():
    out = l2_normalize(np.array([[3.0, 4.0]]))  # 3-4-5 triangle
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_l2_normalize_handles_zero_vector():
    out = l2_normalize(np.array([[0.0, 0.0, 0.0]]))
    assert np.array_equal(out, np.zeros((1, 3), dtype=np.float32))
    assert not np.isnan(out).any()


def test_search_returns_best_match_first():
    store = VectorStore(dim=3)
    store.add("x", np.array([1.0, 0.0, 0.0]))
    store.add("y", np.array([0.0, 1.0, 0.0]))
    store.add("z", np.array([0.9, 0.1, 0.0]))

    results = store.search(np.array([1.0, 0.0, 0.0]), k=3)
    ids = [item_id for item_id, _ in results]
    assert ids[0] == "x"          # exact match wins
    assert ids[1] == "z"          # closest neighbour second
    assert results[0][1] == pytest.approx(1.0)


def test_search_clamps_k_to_store_size():
    store = VectorStore(dim=2)
    store.add("only", np.array([1.0, 1.0]))
    assert len(store.search(np.array([1.0, 1.0]), k=10)) == 1


def test_search_on_empty_store_returns_empty():
    assert VectorStore(dim=4).search(np.array([1.0, 0, 0, 0]), k=3) == []


def test_add_rejects_wrong_dimension():
    store = VectorStore(dim=3)
    with pytest.raises(ValueError):
        store.add("bad", np.array([1.0, 2.0]))
