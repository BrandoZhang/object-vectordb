"""Spec target: indexed search on 100 K rows < 50 ms.

Quick tier uses 1 K rows; full tier (``-m full``) uses 100 K.
"""

from __future__ import annotations

from itertools import cycle

import numpy as np
import pytest

from .conftest import DEFAULT_DIM


def _run_search(benchmark, col, dim, rounds):
    rng = np.random.default_rng(3)
    queries = rng.standard_normal((rounds * 2, dim), dtype=np.float32)
    counter = cycle(queries)

    def op():
        q = next(counter)
        col.search(q.tolist(), vector_field="vec", limit=10, metric="cosine")

    benchmark.pedantic(op, rounds=rounds, iterations=1, warmup_rounds=3)


def test_search_1k_indexed(benchmark, seeded_collection_factory):
    col = seeded_collection_factory(1_000, DEFAULT_DIM, with_index=True)
    _run_search(benchmark, col, DEFAULT_DIM, rounds=30)


def test_search_1k_bruteforce(benchmark, seeded_collection_factory):
    col = seeded_collection_factory(1_000, DEFAULT_DIM, with_index=False)
    _run_search(benchmark, col, DEFAULT_DIM, rounds=30)


@pytest.mark.full
def test_search_100k_indexed(benchmark, seeded_collection_factory):
    col = seeded_collection_factory(100_000, DEFAULT_DIM, with_index=True)
    _run_search(benchmark, col, DEFAULT_DIM, rounds=20)
