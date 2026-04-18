"""Spec target: add + get roundtrip < 10 ms for a single object."""

from __future__ import annotations

from itertools import count, cycle

import numpy as np

from .conftest import DEFAULT_DIM, random_vectors


def test_add_single(benchmark, empty_collection):
    col = empty_collection
    col.register_vector_field("v", dim=DEFAULT_DIM)
    vecs = random_vectors(10_000, DEFAULT_DIM)
    counter = count()

    def op():
        i = next(counter)
        col.add(f"obj_{i:08d}", properties={"bucket": i % 10}, vectors={"v": vecs[i].tolist()})

    benchmark.pedantic(op, rounds=30, iterations=1, warmup_rounds=3)


def test_get_single(benchmark, empty_collection):
    col = empty_collection
    col.register_vector_field("v", dim=DEFAULT_DIM)
    vecs = random_vectors(1000, DEFAULT_DIM)
    for i in range(1000):
        col.add(f"obj_{i:08d}", properties={"bucket": i % 10}, vectors={"v": vecs[i].tolist()})

    rng = np.random.default_rng(7)
    ids = [f"obj_{int(rng.integers(0, 1000)):08d}" for _ in range(1000)]
    counter = cycle(ids)

    def op():
        col.get(next(counter))

    benchmark.pedantic(op, rounds=50, iterations=1, warmup_rounds=3)


def test_add_get_roundtrip(benchmark, empty_collection):
    col = empty_collection
    col.register_vector_field("v", dim=DEFAULT_DIM)
    vecs = random_vectors(10_000, DEFAULT_DIM)
    counter = count()

    def op():
        i = next(counter)
        oid = f"rt_{i:08d}"
        col.add(oid, properties={"bucket": i % 10}, vectors={"v": vecs[i].tolist()})
        col.get(oid)

    benchmark.pedantic(op, rounds=30, iterations=1, warmup_rounds=3)
