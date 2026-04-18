from __future__ import annotations

import pytest

from object_vectordb import ObjectVectorDB


@pytest.fixture
def store(tmp_path):
    """A default Collection at a fresh tmp URI.

    Named `store` for historical reasons — it's a `Collection`, not the
    `ObjectVectorDB` handle. All per-object operations (add, get, search,
    register_vector_field, etc.) live on this fixture.
    """
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    return db.collection("objects")


@pytest.fixture
def db(tmp_path):
    """A fresh `ObjectVectorDB` handle. Use when the test needs multi-collection access."""
    return ObjectVectorDB(uri=str(tmp_path / "db"))
