"""Spec target: export_vectors over 100 K rows < 10 s.

Quick tier uses 1 K rows; full tier (``-m full``) uses 100 K.
"""

from __future__ import annotations

import pytest

from .conftest import DEFAULT_DIM


def test_export_1k(benchmark, seeded_collection_factory):
    col = seeded_collection_factory(1_000, DEFAULT_DIM, with_index=False)

    def op():
        ids, arr = col.export_vectors("vec")
        assert arr.shape == (1_000, DEFAULT_DIM)

    benchmark.pedantic(op, rounds=5, iterations=1, warmup_rounds=1)


@pytest.mark.full
def test_export_100k(benchmark, seeded_collection_factory):
    col = seeded_collection_factory(100_000, DEFAULT_DIM, with_index=False)

    def op():
        ids, arr = col.export_vectors("vec")
        assert arr.shape == (100_000, DEFAULT_DIM)

    benchmark.pedantic(op, rounds=3, iterations=1, warmup_rounds=1)
