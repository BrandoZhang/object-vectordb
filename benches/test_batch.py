"""Spec target: batch_update 1 000 objects < 5 s.

NOTE (observed 2026-04): the current `batch_update` does a per-row
`count_rows("object_id = 'x'")` existence pre-check (backend.py). Since there
is no scalar index on `object_id`, that's O(N) per row → O(N²) overall, and
the 1 000-row case blows the 5 s target by ~60×. Follow-up: create a scalar
index on `object_id` or replace the per-row pre-check with a single batched
query. Tracking in the README Performance section.
"""

from __future__ import annotations

import numpy as np
import pytest

from object_vectordb import ObjectUpdate

from .conftest import DEFAULT_DIM, random_vectors


@pytest.fixture
def prefilled_1k(empty_collection):
    col = empty_collection
    col.register_vector_field("v", dim=DEFAULT_DIM)
    vecs = random_vectors(1000, DEFAULT_DIM)
    for i in range(1000):
        col.add(f"obj_{i:08d}", properties={"bucket": i % 10}, vectors={"v": vecs[i].tolist()})
    return col


@pytest.mark.full
def test_batch_update_1k(benchmark, prefilled_1k):
    col = prefilled_1k
    rng = np.random.default_rng(2)
    updates = [
        ObjectUpdate(
            object_id=f"obj_{i:08d}",
            properties={"bucket": int(rng.integers(0, 100))},
        )
        for i in range(1000)
    ]

    def op():
        col.batch_update(updates)

    benchmark.pedantic(op, rounds=2, iterations=1, warmup_rounds=0)


def test_add_many_1k(benchmark, empty_collection):
    col = empty_collection
    col.register_vector_field("v", dim=DEFAULT_DIM)
    vecs = random_vectors(1000, DEFAULT_DIM)
    round_counter = iter(range(100))

    def op():
        round_idx = next(round_counter)
        items = [
            {
                "object_id": f"r{round_idx}_obj_{i:04d}",
                "properties": {"bucket": i % 10},
                "vectors": {"v": vecs[i].tolist()},
            }
            for i in range(1000)
        ]
        col.add_many(items)

    benchmark.pedantic(op, rounds=2, iterations=1, warmup_rounds=0)
