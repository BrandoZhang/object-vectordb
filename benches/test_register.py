"""Spec target: register_vector_field on a 1 M-row table < 1 s (zero-copy).

Quick tier uses 10 K rows; full tier (``-m full``) uses 1 M.
"""

from __future__ import annotations

import pytest

from .conftest import DEFAULT_DIM


def test_register_10k(benchmark, seeded_collection_factory):
    # Baseline: a collection already seeded to 10k rows with one vector field.
    # The bench measures adding ANOTHER vector field on top (the zero-copy path).
    col = seeded_collection_factory(10_000, DEFAULT_DIM, with_index=False)
    counter = iter(range(100))

    def op():
        i = next(counter)
        col.register_vector_field(f"extra_{i}", dim=256)

    benchmark.pedantic(op, rounds=20, iterations=1, warmup_rounds=1)


@pytest.mark.full
def test_register_1m(benchmark, seeded_collection_factory):
    col = seeded_collection_factory(1_000_000, DEFAULT_DIM, with_index=False)
    counter = iter(range(20))

    def op():
        i = next(counter)
        col.register_vector_field(f"extra_{i}", dim=256)

    benchmark.pedantic(op, rounds=10, iterations=1, warmup_rounds=1)
