from __future__ import annotations

import math

import pytest

from object_store import DimensionMismatch, MetricMismatch, VectorFieldNotRegistered


def _setup_known(store, vectors):
    store.register_vector_field("v", dim=3)
    for i, vec in enumerate(vectors):
        store.add(f"id{i}", properties={"n": i}, vectors={"v": vec})


def test_search_returns_top_k_by_similarity_cosine(store):
    # Query aligns with id0, orthogonal to id1, opposite to id2
    _setup_known(store, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    hits = store.search([1.0, 0.0, 0.0], vector_field="v", limit=3, metric="cosine")
    assert [h.object_id for h in hits] == ["id0", "id1", "id2"]
    # Scores sorted descending
    assert hits[0].score > hits[1].score > hits[2].score
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[1].score == pytest.approx(0.0, abs=1e-5)
    assert hits[2].score == pytest.approx(-1.0, abs=1e-5)


def test_search_l2_monotonic(store):
    _setup_known(store, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    hits = store.search([1.0, 0.0, 0.0], vector_field="v", limit=3, metric="l2")
    # Closest to [1,0,0] is id0 (distance 0), then id2 (distance 0.25), then id1 (distance 1)
    assert [h.object_id for h in hits] == ["id0", "id2", "id1"]
    assert 0 < hits[2].score <= hits[1].score <= hits[0].score <= 1.0


def test_search_dot_score_recovers_raw_dot(store):
    _setup_known(store, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    hits = store.search([1.0, 0.0, 0.0], vector_field="v", limit=3, metric="dot")
    # Raw dots: id0=1, id1=2, id2=0.5. Sorted desc: id1, id0, id2.
    assert [h.object_id for h in hits] == ["id1", "id0", "id2"]
    assert hits[0].score == pytest.approx(2.0, abs=1e-5)
    assert hits[1].score == pytest.approx(1.0, abs=1e-5)
    assert hits[2].score == pytest.approx(0.5, abs=1e-5)


def test_search_with_where_filter(store):
    _setup_known(store, [[1.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.8, 0.0, 0.0]])
    hits = store.search([1.0, 0.0, 0.0], vector_field="v", limit=10, where="n >= 1")
    ids = [h.object_id for h in hits]
    assert "id0" not in ids
    assert set(ids) == {"id1", "id2"}


def test_search_select_returns_only_requested_props(store):
    store.register_vector_field("v", dim=2)
    store.add("x", properties={"a": 1, "b": 2, "c": 3}, vectors={"v": [1.0, 0.0]})
    hits = store.search([1.0, 0.0], vector_field="v", limit=1, select=["a", "c"])
    assert hits[0].properties == {"a": 1, "c": 3}


def test_search_default_metric_is_cosine(store):
    _setup_known(store, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hits = store.search([1.0, 0.0, 0.0], vector_field="v", limit=2)
    assert hits[0].object_id == "id0"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_unregistered_field_raises(store):
    with pytest.raises(VectorFieldNotRegistered):
        store.search([1.0, 0.0], vector_field="nope", limit=1)


def test_search_empty_query_raises(store):
    store.register_vector_field("v", dim=3)
    with pytest.raises(ValueError):
        store.search([], vector_field="v", limit=1)


def test_search_wrong_dim_raises(store):
    store.register_vector_field("v", dim=3)
    with pytest.raises(DimensionMismatch):
        store.search([1.0, 2.0], vector_field="v", limit=1)


def test_search_without_index_still_works(store):
    _setup_known(store, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hits = store.search([1.0, 0.0, 0.0], vector_field="v", limit=2, metric="cosine")
    assert len(hits) == 2


def test_search_score_values_are_finite(store):
    store.register_vector_field("v", dim=2)
    store.add("a", vectors={"v": [1.0, 0.0]})
    hits = store.search([1.0, 0.0], vector_field="v", limit=1, metric="l2")
    assert math.isfinite(hits[0].score)


def test_metric_mismatch_with_existing_index_raises(store):
    store.register_vector_field("v", dim=3)
    for i in range(300):
        # enough rows for IVF index
        store.add(f"id{i}", vectors={"v": [float(i), 0.0, 0.0]})
    store.create_index("v", index_type="IVF_PQ", metric="cosine", num_partitions=2, num_sub_vectors=1)
    with pytest.raises(MetricMismatch):
        store.search([1.0, 0.0, 0.0], vector_field="v", limit=1, metric="l2")
