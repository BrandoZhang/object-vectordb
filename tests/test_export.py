from __future__ import annotations

import numpy as np
import pytest

from object_vectordb import VectorFieldNotRegistered


def test_export_vectors_returns_aligned_arrays(store):
    store.register_vector_field("v", dim=3)
    for i in range(5):
        store.add(f"id{i}", vectors={"v": [float(i), 0.0, 0.0]})
    ids, arr = store.export_vectors("v")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (5, 3)
    assert len(ids) == 5
    # Each id maps to its vector
    lookup = dict(zip(ids, arr.tolist()))
    for i in range(5):
        assert lookup[f"id{i}"] == pytest.approx([float(i), 0.0, 0.0])


def test_export_skips_null_vectors(store):
    store.register_vector_field("v", dim=2)
    store.add("a", vectors={"v": [1.0, 2.0]})
    store.add("b", properties={"has_vec": False})  # v is null here
    store.add("c", vectors={"v": [3.0, 4.0]})
    ids, arr = store.export_vectors("v")
    assert set(ids) == {"a", "c"}
    assert arr.shape == (2, 2)


def test_export_with_where_filter(store):
    store.register_vector_field("v", dim=2)
    for i in range(5):
        store.add(f"id{i}", properties={"n": i}, vectors={"v": [float(i), 0.0]})
    ids, arr = store.export_vectors("v", where="n >= 3")
    assert set(ids) == {"id3", "id4"}
    assert arr.shape == (2, 2)


def test_export_empty(store):
    store.register_vector_field("v", dim=3)
    ids, arr = store.export_vectors("v")
    assert ids == []
    assert arr.shape == (0, 3)


def test_export_unregistered_raises(store):
    with pytest.raises(VectorFieldNotRegistered):
        store.export_vectors("ghost")
