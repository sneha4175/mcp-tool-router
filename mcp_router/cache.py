"""Query -> tool-set result cache (v0.3).

Retrieval is the expensive part of a request: embedding the query and searching
the vector store. But real traffic is repetitive - the same handful of phrasings
recur constantly ("read a file", "convert currency", a client retrying). Redoing
the full embed + search for an identical query is wasted work.

This module memoizes the top-k result for a query so a repeat skips both the
embedding call and the vector search and returns the previously computed tool
set. It sits as a thin layer *in front of* the retriever (see
:class:`~mcp_router.registry.ToolRegistry.retrieve`); the retriever itself stays
a pure, cache-unaware component.

Two bounds keep the cache honest:

* **TTL** - every entry has an expiry. A hit past its TTL is treated as a miss
  and recomputed, so results can't go stale indefinitely.
* **LRU max-size** - the cache holds at most ``max_entries``; inserting beyond
  that evicts the least-recently-used entry. This caps memory no matter how many
  distinct queries arrive.

The cache is also cleared wholesale whenever the tool registry changes (upstreams
reconnect / tools refresh), because a cached tool set computed against the old
catalogue may no longer be correct.

The clock is injectable so TTL behaviour is testable deterministically without
``sleep``.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

# The cache key: a normalized query string plus k. Two keys that only differ in
# casing or surrounding whitespace should hit the same entry, so we fold those
# out here. k is part of the key because "top-2" and "top-5" are different
# answers to the same words.
CacheKey = Tuple[str, int]


def normalize_query(query: str, k: int) -> CacheKey:
    """Build the canonical cache key for a ``(query, k)`` pair.

    Lowercasing and stripping means ``"Read A File"`` and ``"  read a file "``
    map to the same entry - the common, cheap normalizations that don't change
    intent. We deliberately stop there: collapsing internal whitespace or
    stemming would risk merging queries that are meaningfully different.
    """
    return (query.strip().lower(), int(k))


class QueryCache:
    """A TTL + LRU cache mapping a normalized query key to a tool-set result.

    Not thread-safe by design: the gateway handles one request at a time per
    worker, and adding locking here would be complexity without a demonstrated
    need. If a future async/multi-threaded path shares one cache, wrap the
    ``get``/``set`` calls in a lock then.
    """

    def __init__(
        self,
        enabled: bool = True,
        ttl_seconds: float = 300.0,
        max_entries: int = 512,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self.enabled = enabled
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        # monotonic() (not time()) so TTLs are immune to wall-clock jumps (NTP,
        # DST). Injectable purely so tests can advance time deterministically.
        self._clock = clock

        # Ordered by recency: the left end is the least-recently-used entry (the
        # next eviction victim), the right end the most-recently-used. Each value
        # is (expires_at, result).
        self._store: "OrderedDict[CacheKey, Tuple[float, Any]]" = OrderedDict()

        # Observability counters. hits/misses drive the /stats hit-rate;
        # evictions and invalidations help explain a low hit rate ("too small"
        # vs "registry keeps changing").
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0

    def get(self, key: CacheKey) -> Optional[Any]:
        """Return the cached value for ``key`` or ``None`` on a miss.

        A miss covers three cases, all counted as a miss: never-seen key, and an
        expired entry (which is dropped here so it can't linger). On a hit the
        entry is marked most-recently-used so the LRU order stays accurate.
        """
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None

        expires_at, value = entry
        if expires_at <= self._clock():
            # Expired: evict and treat as a miss so the caller recomputes.
            del self._store[key]
            self.misses += 1
            return None

        # Fresh hit - bump recency and return.
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: CacheKey, value: Any) -> None:
        """Insert/refresh ``key`` and evict the LRU entry if over capacity."""
        expires_at = self._clock() + self.ttl_seconds
        self._store[key] = (expires_at, value)
        self._store.move_to_end(key)

        # Enforce the size bound. A single set adds one entry, but loop anyway so
        # the invariant holds even if max_entries were lowered at runtime.
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)  # pop the least-recently-used
            self.evictions += 1

    def clear(self) -> None:
        """Drop every entry (used to invalidate on a registry change).

        Counters other than ``invalidations`` are preserved: hit/miss history is
        a lifetime metric, not tied to the current entry set.
        """
        if self._store:
            self.invalidations += 1
        self._store.clear()

    def stats(self) -> Dict[str, Any]:
        """A JSON-serializable snapshot for the ``/stats`` endpoint."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total else 0.0
        return {
            "enabled": self.enabled,
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
            "evictions": self.evictions,
            "invalidations": self.invalidations,
        }
