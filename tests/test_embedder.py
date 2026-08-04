"""Unit tests for the embedder."""

from __future__ import annotations

import numpy as np

from mcp_router.embedder import HashingEmbedder, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("Read_File from Disk!") == ["read", "file", "from", "disk"]


def test_embedding_shape_and_dtype():
    emb = HashingEmbedder(dim=64)
    vecs = emb.embed(["hello world", "another string"])
    assert vecs.shape == (2, 64)
    assert vecs.dtype == np.float32


def test_embedding_is_deterministic():
    # Same text -> identical vector, even across separate embedder instances.
    a = HashingEmbedder(dim=128).embed_one("convert currency amount")
    b = HashingEmbedder(dim=128).embed_one("convert currency amount")
    assert np.array_equal(a, b)


def test_shared_vocabulary_scores_higher_than_unrelated():
    emb = HashingEmbedder(dim=512)
    base = emb.embed_one("send an email message to a recipient")
    similar = emb.embed_one("send email to recipient")
    unrelated = emb.embed_one("weather forecast temperature for a city")

    def cosine(x, y):
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    assert cosine(base, similar) > cosine(base, unrelated)


def test_empty_text_gives_zero_vector():
    emb = HashingEmbedder(dim=32)
    assert not emb.embed_one("").any()
