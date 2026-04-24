"""Fixtures and helpers for the benchmark suite.

The suite runs against ephemeral temp dirs. Seeding is the slowest step, so
the `seeded_collection` fixture caches datasets per (size, dim, indexed)
tuple within a pytest session — `pytest-benchmark` calls bench functions
multiple times for statistical rigor, and reseeding on every call would
blow up the runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from object_vectordb import Collection, ObjectAdd, ObjectVectorDB

DEFAULT_DIM = 1024
SEED = 42


def random_vectors(n: int, dim: int, seed: int = SEED) -> np.ndarray:
    """Reproducible random float32 vectors, shape (n, dim)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim), dtype=np.float32)


def _seed_collection(
    collection: Collection,
    n: int,
    dim: int,
    *,
    with_index: bool,
    field: str = "vec",
) -> None:
    collection.register_vector_field(field, dim=dim)
    vectors = random_vectors(n, dim)
    # Batch inserts via batch_add for speed. Chunks of 5 000 rows keep pyarrow
    # RecordBatch sizes reasonable.
    chunk = 5000
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        items = [
            ObjectAdd(
                object_id=f"obj_{i:08d}",
                properties={"bucket": i % 10, "tag": f"t{i % 100}"},
                vectors={field: vectors[i].tolist()},
            )
            for i in range(start, end)
        ]
        collection.batch_add(items)
    if with_index and n >= 256:
        # num_partitions must be <= n / some threshold. Use 16 for small N,
        # scale roughly with sqrt(n) for larger.
        num_partitions = max(2, min(256, int(n**0.5)))
        num_sub_vectors = max(1, min(16, dim // 8))
        collection.create_index(
            field,
            index_type="IVF_PQ",
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
        )


@pytest.fixture(scope="session")
def dataset_cache(tmp_path_factory) -> dict[tuple, Path]:
    """Session-scoped mapping (size, dim, indexed) → uri of a seeded DB."""
    return {}


@pytest.fixture(scope="session")
def seeded_collection_factory(tmp_path_factory, dataset_cache):
    """Factory that returns a seeded Collection, caching by (n, dim, index)."""

    def _make(n: int, dim: int = DEFAULT_DIM, *, with_index: bool = False) -> Collection:
        key = (n, dim, with_index)
        if key not in dataset_cache:
            uri = tmp_path_factory.mktemp(f"bench_{n}_{dim}_{int(with_index)}")
            db = ObjectVectorDB(uri=str(uri))
            col = db.collection("bench")
            _seed_collection(col, n, dim, with_index=with_index)
            dataset_cache[key] = str(uri)
        # Re-open each call so the caller gets a fresh Collection handle.
        db = ObjectVectorDB(uri=dataset_cache[key])
        return db.collection("bench")

    return _make


@pytest.fixture
def empty_collection(tmp_path) -> Collection:
    """A fresh empty Collection at a tmp URI. Per-test; not cached."""
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    return db.collection("bench")


def random_update_batch(collection: Collection, n: int, dim: int) -> list[dict[str, Any]]:
    """Pre-build N ObjectUpdate-shaped dicts for batch_update timing."""
    rng = np.random.default_rng(SEED + 1)
    return [
        {
            "object_id": f"obj_{i:08d}",
            "properties": {"bucket": int(rng.integers(0, 100))},
        }
        for i in range(n)
    ]
