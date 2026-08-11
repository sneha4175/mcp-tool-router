"""Tests for the v0.3 query-result cache.

Two layers are exercised:

* the :class:`QueryCache` unit in isolation (TTL, LRU, stats) using an injected
  fake clock so time-based behaviour is deterministic - no ``sleep``;
* the cache wired into :class:`ToolRegistry`, proving a repeated query skips
  recomputation, that a registry change invalidates cached results, and that the
  hit/miss stats are exposed correctly.

Everything runs offline on the deterministic hashing embedder (conftest forces
``MCP_ROUTER_EMBEDDER=hashing``).
"""

from __future__ import annotations

import pytest

from mcp_router.cache import QueryCache, normalize_query
from mcp_router.config import parse_config
from mcp_router.embedder import HashingEmbedder
from mcp_router.registry import ToolRegistry

from .conftest import SAMPLE_CONFIG


class FakeClock:
    """A hand-cranked monotonic clock so TTL tests don't sleep."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _count_calls(registry: ToolRegistry) -> dict:
    """Wrap the registry's retriever.retrieve with a call counter (a spy).

    The registry caches in front of ``self.retriever.retrieve``, so this counter
    increments only when a query actually reaches the (expensive) embed+search
    path - i.e. on a miss. A cache hit must NOT bump it.
    """
    counter = {"n": 0}
    original = registry.retriever.retrieve

    def spy(query, k):
        counter["n"] += 1
        return original(query, k)

    registry.retriever.retrieve = spy  # type: ignore[method-assign]
    return counter


# --------------------------------------------------------------------------- #
# QueryCache unit tests
# --------------------------------------------------------------------------- #

def test_get_miss_then_hit():
    cache = QueryCache(ttl_seconds=100, max_entries=8, clock=FakeClock())
    key = normalize_query("read a file", 3)

    assert cache.get(key) is None          # miss - nothing stored yet
    cache.set(key, ["result"])
    assert cache.get(key) == ["result"]    # hit
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_normalize_query_folds_case_and_whitespace():
    assert normalize_query("  Read A File ", 3) == normalize_query("read a file", 3)
    # k is part of the key: same words, different k => different entry.
    assert normalize_query("read a file", 3) != normalize_query("read a file", 5)


def test_ttl_expiry_recomputes():
    clock = FakeClock()
    cache = QueryCache(ttl_seconds=10, max_entries=8, clock=clock)
    key = normalize_query("convert currency", 2)

    cache.set(key, ["v"])
    assert cache.get(key) == ["v"]         # still fresh

    clock.advance(11)                      # past the TTL
    assert cache.get(key) is None          # expired -> miss (must recompute)
    assert cache.stats()["misses"] == 1


def test_lru_evicts_oldest():
    cache = QueryCache(ttl_seconds=1000, max_entries=2, clock=FakeClock())
    a, b, c = (("a", 1), ("b", 1), ("c", 1))

    cache.set(a, ["a"])
    cache.set(b, ["b"])
    cache.get(a)               # touch 'a' -> 'b' is now the least-recently-used
    cache.set(c, ["c"])        # over capacity -> evict 'b'

    assert cache.get(b) is None    # evicted
    assert cache.get(a) == ["a"]   # survived (was recently used)
    assert cache.get(c) == ["c"]   # newest
    assert cache.stats()["evictions"] == 1
    assert cache.stats()["size"] == 2


def test_clear_counts_invalidation_but_keeps_hit_history():
    cache = QueryCache(ttl_seconds=100, max_entries=8, clock=FakeClock())
    key = normalize_query("read a file", 1)
    cache.set(key, ["v"])
    cache.get(key)                 # 1 hit

    cache.clear()
    assert cache.get(key) is None  # gone after invalidation
    stats = cache.stats()
    assert stats["invalidations"] == 1
    assert stats["hits"] == 1      # lifetime history preserved
    assert stats["size"] == 0


def test_hit_rate_reported():
    cache = QueryCache(ttl_seconds=100, max_entries=8, clock=FakeClock())
    key = normalize_query("x", 1)
    cache.set(key, ["v"])
    cache.get(key)   # hit
    cache.get(key)   # hit
    cache.get(normalize_query("y", 1))  # miss
    assert cache.stats()["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        QueryCache(ttl_seconds=0)
    with pytest.raises(ValueError):
        QueryCache(max_entries=0)


# --------------------------------------------------------------------------- #
# Registry integration tests
# --------------------------------------------------------------------------- #

def test_cache_hit_skips_recompute(registry: ToolRegistry):
    """A repeated (normalized-identical) query must not hit the retriever again."""
    calls = _count_calls(registry)

    first = registry.retrieve("convert currency", 2)   # miss -> computes
    second = registry.retrieve("  Convert currency ", 2)  # hit (case+trim normalized)

    assert calls["n"] == 1, "second identical query must be served from cache"
    assert [t.name for t, _ in first] == [t.name for t, _ in second]


def test_different_k_is_a_separate_entry(registry: ToolRegistry):
    calls = _count_calls(registry)
    registry.retrieve("read a file", 1)   # miss
    registry.retrieve("read a file", 3)   # miss - different k, not a hit
    assert calls["n"] == 2


def test_ttl_expiry_via_registry(registry: ToolRegistry):
    clock = FakeClock()
    registry._cache = QueryCache(enabled=True, ttl_seconds=5, max_entries=16, clock=clock)
    calls = _count_calls(registry)

    registry.retrieve("read a file", 1)   # miss -> compute (n=1)
    registry.retrieve("read a file", 1)   # hit  -> no compute (n=1)
    assert calls["n"] == 1

    clock.advance(6)                       # entry now past its TTL
    registry.retrieve("read a file", 1)   # expired -> recompute (n=2)
    assert calls["n"] == 2


def test_refresh_invalidates_cache(registry: ToolRegistry):
    """A registry change (refresh) must drop cached tool sets."""
    registry.retrieve("convert currency", 2)          # populate
    assert registry.stats()["cache"]["size"] == 1

    registry.refresh()                                 # tools re-discovered
    assert registry.stats()["cache"]["size"] == 0
    assert registry.stats()["cache"]["invalidations"] == 1

    # And the very next identical query is a miss again (recomputed), not a hit.
    calls = _count_calls(registry)
    registry.retrieve("convert currency", 2)
    assert calls["n"] == 1


def test_stats_reflect_hits_and_misses(registry: ToolRegistry):
    registry.retrieve("convert currency", 2)   # miss
    registry.retrieve("convert currency", 2)   # hit
    registry.retrieve("read a file", 1)        # miss

    cache_stats = registry.stats()["cache"]
    assert cache_stats["hits"] == 1
    assert cache_stats["misses"] == 2
    assert cache_stats["enabled"] is True
    assert registry.stats()["tools"] == 6


def test_disabled_cache_passes_through():
    config = parse_config({**SAMPLE_CONFIG, "cache": {"enabled": False}})
    registry = ToolRegistry(config, embedder=HashingEmbedder(dim=256))
    calls = _count_calls(registry)

    registry.retrieve("convert currency", 2)   # recomputes
    registry.retrieve("convert currency", 2)   # recomputes again - no caching

    assert calls["n"] == 2
    assert registry.stats()["cache"]["enabled"] is False


def test_empty_query_is_not_cached(registry: ToolRegistry):
    assert registry.retrieve("   ", 5) == []
    # Whitespace-only queries bypass the cache entirely, so nothing is stored.
    assert registry.stats()["cache"]["size"] == 0


def test_stats_endpoint_exposes_cache(registry: ToolRegistry):
    from fastapi.testclient import TestClient

    from mcp_router.server import create_app

    client = TestClient(create_app(registry))
    # Drive one miss + one hit through the HTTP path.
    for _ in range(2):
        client.post(
            "/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list",
                  "params": {"query": "weather forecast", "k": 1}},
        )

    body = client.get("/stats").json()
    assert body["tools"] == 6
    assert body["cache"]["hits"] == 1
    assert body["cache"]["misses"] == 1


def test_config_parses_cache_block():
    config = parse_config(
        {**SAMPLE_CONFIG, "cache": {"enabled": True, "ttl_seconds": 42, "max_entries": 7}}
    )
    assert config.cache.enabled is True
    assert config.cache.ttl_seconds == 42.0
    assert config.cache.max_entries == 7


def test_config_cache_defaults_when_absent():
    config = parse_config(SAMPLE_CONFIG)  # no 'cache' key
    assert config.cache.enabled is True
    assert config.cache.ttl_seconds == 300.0
    assert config.cache.max_entries == 512
