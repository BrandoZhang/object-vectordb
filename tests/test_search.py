from __future__ import annotations

import math

import pytest

from object_vectordb import DimensionMismatch, MetricMismatch, VectorFieldNotRegistered


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
    store.create_index(
        "v", index_type="IVF_PQ", metric="cosine", num_partitions=2, num_sub_vectors=1
    )
    with pytest.raises(MetricMismatch):
        store.search([1.0, 0.0, 0.0], vector_field="v", limit=1, metric="l2")


# ---------------------------------------------------------------------------
# search_within (radius / distance-bounded search)
# ---------------------------------------------------------------------------


def test_search_within_cosine_returns_all_within_radius(store):
    # id0 identical (d=0), id1 nearly-aligned (d≈0.006), id2 orthogonal (d=1),
    # id3 opposite (d=2). Radius 0.1 should return only id0 and id1.
    _setup_known(
        store,
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    )
    hits = store.search_within([1.0, 0.0, 0.0], vector_field="v", max_distance=0.1, metric="cosine")
    assert [h.object_id for h in hits] == ["id0", "id1"]
    assert hits[0].score > hits[1].score


def test_search_within_l2_returns_all_within_radius(store):
    # Squared-L2 distances to [0,0,0]: 0, 1, 4, 9. max_distance=2 → id0, id1.
    _setup_known(
        store,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    )
    hits = store.search_within([0.0, 0.0, 0.0], vector_field="v", max_distance=2.0, metric="l2")
    assert [h.object_id for h in hits] == ["id0", "id1"]


def test_search_within_dot_returns_all_within_radius(store):
    # Query [1,0,0]. Raw dots: id0=1, id1=2, id2=0.5, id3=-1.
    # LanceDB distance = 1 - dot: id0=0, id1=-1, id2=0.5, id3=2.
    # max_distance=0.6 → {id0, id1, id2}, sorted by ascending distance.
    _setup_known(
        store,
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    )
    hits = store.search_within([1.0, 0.0, 0.0], vector_field="v", max_distance=0.6, metric="dot")
    assert [h.object_id for h in hits] == ["id1", "id0", "id2"]
    # Score equals raw dot for metric="dot".
    assert hits[0].score == pytest.approx(2.0, abs=1e-5)


def test_search_within_empty_when_radius_too_tight(store):
    _setup_known(store, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # Query is orthogonal to both stored vectors (cosine distance = 1); a very
    # tight radius returns nothing.
    hits = store.search_within(
        [1.0, 0.0, 0.0], vector_field="v", max_distance=0.01, metric="cosine"
    )
    assert hits == []


def test_search_within_min_distance_excludes_inner_band(store):
    # Stored vectors at squared-L2 distances 0, 1, 4, 9 from the query origin.
    _setup_known(
        store,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    )
    # "similar but not identical": exclude the exact match (d=0).
    hits = store.search_within(
        [0.0, 0.0, 0.0],
        vector_field="v",
        max_distance=5.0,
        min_distance=0.5,
        metric="l2",
    )
    assert [h.object_id for h in hits] == ["id1", "id2"]


def test_search_within_with_where_filter(store):
    _setup_known(store, [[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.9, 0.1, 0.0]])
    hits = store.search_within(
        [1.0, 0.0, 0.0],
        vector_field="v",
        max_distance=0.5,
        metric="cosine",
        where="n >= 1",
    )
    ids = {h.object_id for h in hits}
    assert "id0" not in ids
    assert ids == {"id1", "id2"}


def test_search_within_with_select(store):
    store.register_vector_field("v", dim=2)
    store.add("x", properties={"a": 1, "b": 2, "c": 3}, vectors={"v": [1.0, 0.0]})
    hits = store.search_within([1.0, 0.0], vector_field="v", max_distance=0.1, select=["a", "c"])
    assert len(hits) == 1
    assert hits[0].properties == {"a": 1, "c": 3}


def test_search_within_results_sorted_by_score_desc(store):
    _setup_known(
        store,
        [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.6, 0.4, 0.0], [0.4, 0.6, 0.0]],
    )
    hits = store.search_within([1.0, 0.0, 0.0], vector_field="v", max_distance=0.5, metric="cosine")
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_within_unregistered_field_raises(store):
    with pytest.raises(VectorFieldNotRegistered):
        store.search_within([1.0, 0.0], vector_field="nope", max_distance=0.1)


def test_search_within_wrong_dim_raises(store):
    store.register_vector_field("v", dim=3)
    with pytest.raises(DimensionMismatch):
        store.search_within([1.0, 2.0], vector_field="v", max_distance=0.1)


def test_search_within_empty_query_raises(store):
    store.register_vector_field("v", dim=3)
    with pytest.raises(ValueError):
        store.search_within([], vector_field="v", max_distance=0.1)


@pytest.mark.parametrize(
    "max_distance,min_distance",
    [
        (float("nan"), None),
        (float("inf"), None),
        (0.5, float("nan")),
        (0.5, 0.5),  # min == max
        (0.5, 0.8),  # min > max
    ],
)
def test_search_within_invalid_bounds_raise(store, max_distance, min_distance):
    store.register_vector_field("v", dim=3)
    store.add("a", vectors={"v": [1.0, 0.0, 0.0]})
    with pytest.raises(ValueError):
        store.search_within(
            [1.0, 0.0, 0.0],
            vector_field="v",
            max_distance=max_distance,
            min_distance=min_distance,
        )


def test_search_within_limit_caps_results(store):
    _setup_known(store, [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.98, 0.02, 0.0]])
    # All three are well within the radius; limit=2 caps the returned list.
    hits = store.search_within(
        [1.0, 0.0, 0.0], vector_field="v", max_distance=0.5, limit=2, metric="cosine"
    )
    assert len(hits) == 2


def test_search_within_default_metric_is_cosine(store):
    _setup_known(store, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hits = store.search_within([1.0, 0.0, 0.0], vector_field="v", max_distance=0.1)
    assert [h.object_id for h in hits] == ["id0"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_within_metric_mismatch_with_index_raises(store):
    store.register_vector_field("v", dim=3)
    for i in range(300):
        store.add(f"id{i}", vectors={"v": [float(i), 0.0, 0.0]})
    store.create_index(
        "v", index_type="IVF_PQ", metric="cosine", num_partitions=2, num_sub_vectors=1
    )
    with pytest.raises(MetricMismatch):
        store.search_within([1.0, 0.0, 0.0], vector_field="v", max_distance=0.5, metric="l2")


def test_search_within_exact_mode_returns_all_matches(store):
    _setup_known(
        store,
        [[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    )
    hits = store.search_within(
        [1.0, 0.0, 0.0],
        vector_field="v",
        max_distance=0.5,
        metric="cosine",
        exact=True,
    )
    assert {h.object_id for h in hits} == {"id0", "id1"}
