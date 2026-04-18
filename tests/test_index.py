from __future__ import annotations

import pytest

from object_vectordb import SchemaError, VectorFieldNotRegistered


def _seed(store, n=300, dim=4):
    store.register_vector_field("v", dim=dim)
    for i in range(n):
        vec = [float(i % 3 == 0), float(i % 3 == 1), float(i % 3 == 2), float(i)]
        store.add(f"id{i}", properties={"n": i}, vectors={"v": vec})


def test_create_index_sets_has_index(store):
    _seed(store, n=300)
    store.create_index(
        "v", index_type="IVF_PQ", metric="cosine", num_partitions=2, num_sub_vectors=1
    )
    info = {f.name: f for f in store.vector_fields()}["v"]
    assert info.has_index is True


def test_index_info_returns_registered_params(store):
    _seed(store, n=300)
    store.create_index(
        "v", index_type="IVF_PQ", metric="cosine", num_partitions=2, num_sub_vectors=1
    )
    idx = store.index_info("v")
    assert idx is not None
    assert idx.index_type == "IVF_PQ"
    assert idx.metric == "cosine"
    assert idx.params.get("num_partitions") == 2
    assert idx.params.get("num_sub_vectors") == 1


def test_index_info_none_when_no_index(store):
    store.register_vector_field("v", dim=3)
    assert store.index_info("v") is None


def test_drop_index_clears_has_index(store):
    _seed(store, n=300)
    store.create_index(
        "v", index_type="IVF_PQ", metric="cosine", num_partitions=2, num_sub_vectors=1
    )
    store.drop_index("v")
    info = {f.name: f for f in store.vector_fields()}["v"]
    assert info.has_index is False
    assert store.index_info("v") is None


def test_rebuild_index_preserves_config(store):
    _seed(store, n=300)
    store.create_index(
        "v", index_type="IVF_PQ", metric="cosine", num_partitions=2, num_sub_vectors=1
    )
    store.rebuild_index("v")
    idx = store.index_info("v")
    assert idx.index_type == "IVF_PQ"
    assert idx.metric == "cosine"


def test_rebuild_without_prior_raises(store):
    store.register_vector_field("v", dim=3)
    with pytest.raises(SchemaError):
        store.rebuild_index("v")


def test_index_ops_on_unregistered_raise(store):
    with pytest.raises(VectorFieldNotRegistered):
        store.create_index("missing")
    with pytest.raises(VectorFieldNotRegistered):
        store.drop_index("missing")
    with pytest.raises(VectorFieldNotRegistered):
        store.rebuild_index("missing")


def test_search_still_works_after_drop_index(store):
    _seed(store, n=300)
    store.create_index(
        "v", index_type="IVF_PQ", metric="cosine", num_partitions=2, num_sub_vectors=1
    )
    store.drop_index("v")
    hits = store.search([1.0, 0.0, 0.0, 0.0], vector_field="v", limit=5, metric="cosine")
    assert len(hits) == 5
