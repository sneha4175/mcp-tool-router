"""A tiny in-memory cosine-similarity vector store.

For an MVP we do not need FAISS, a vector database, or an ANN index. The number
of tools a gateway fronts is small (tens, maybe low hundreds), so an exact
brute-force cosine search over a single NumPy matrix is both simplest and
fastest. Swapping in an approximate index later is a localized change behind the
same ``add``/``search`` interface.

Design choices:

* Vectors are L2-normalized on insert. Once every stored vector has unit length,
  cosine similarity reduces to a plain dot product, so search is one matrix-
  vector multiply.
* Zero vectors (which have no direction) are kept as all-zeros; they simply
  score 0 against everything, which is the sensible "no signal" behaviour.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` with each row scaled to unit L2 length.

    Rows that are all zero are left as zero (instead of producing NaNs from a
    divide-by-zero).
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero: where the norm is 0 we divide by 1 instead, which
    # leaves the (already zero) row untouched.
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


class VectorStore:
    """Holds id-tagged unit vectors and does exact cosine top-k search."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: List[str] = []
        # Kept as a list of rows and stacked lazily on search so that repeated
        # add() calls during index build stay cheap.
        self._rows: List[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, item_id: str, vector: np.ndarray) -> None:
        """Store one vector under ``item_id`` (normalized on the way in)."""
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"vector has dim {vector.shape[0]}, store expects {self.dim}"
            )
        self._ids.append(item_id)
        self._rows.append(l2_normalize(vector)[0])
        self._matrix = None  # invalidate the cached stacked matrix

    def _ensure_matrix(self) -> np.ndarray:
        if self._matrix is None:
            if self._rows:
                self._matrix = np.vstack(self._rows)
            else:
                self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        return self._matrix

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Return the ``k`` highest-scoring ``(id, cosine_score)`` pairs.

        Results are sorted by score descending. ``k`` is clamped to the number
        of stored items, so asking for more than exist is safe.
        """
        if k <= 0 or len(self._ids) == 0:
            return []

        matrix = self._ensure_matrix()
        q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(-1))[0]

        # Cosine similarity == dot product because everything is unit length.
        scores = matrix @ q

        k = min(k, len(self._ids))
        # argpartition gets the top-k cheaply (O(n)); we then sort just those k.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._ids[i], float(scores[i])) for i in top_idx]
