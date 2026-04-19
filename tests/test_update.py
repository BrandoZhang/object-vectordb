from __future__ import annotations

import pytest

from object_vectordb import ObjectNotFound


def test_update_properties_only_preserves_vectors(store):
    store.register_vector_field("v", dim=3)
    store.add("x", properties={"title": "a"}, vectors={"v": [1.0, 2.0, 3.0]})

    store.update("x", properties={"title": "b"})
    obj = store.get("x")
    assert obj.properties["title"] == "b"
    assert obj.vectors["v"] == pytest.approx([1.0, 2.0, 3.0])


def test_update_vectors_only_preserves_properties(store):
    store.register_vector_field("v", dim=2)
    store.add("x", properties={"title": "a"}, vectors={"v": [1.0, 2.0]})

    store.update("x", vectors={"v": [9.0, 8.0]})
    obj = store.get("x")
    assert obj.properties["title"] == "a"
    assert obj.vectors["v"] == pytest.approx([9.0, 8.0])


def test_update_both(store):
    store.register_vector_field("v", dim=2)
    store.add("x", properties={"a": 1}, vectors={"v": [1.0, 2.0]})

    store.update("x", properties={"a": 99}, vectors={"v": [5.0, 6.0]})
    obj = store.get("x")
    assert obj.properties["a"] == 99
    assert obj.vectors["v"] == pytest.approx([5.0, 6.0])


def test_update_missing_raises(store):
    with pytest.raises(ObjectNotFound):
        store.update("ghost", properties={"a": 1})


def test_clear_property_to_none(store):
    store.add("x", properties={"title": "a", "views": 5})
    store.update("x", properties={"title": None})
    obj = store.get("x")
    assert obj.properties["title"] is None
    assert obj.properties["views"] == 5


def test_clear_vector_to_none(store):
    store.register_vector_field("v", dim=3)
    store.add("x", properties={"title": "a"}, vectors={"v": [1.0, 2.0, 3.0]})
    store.update("x", vectors={"v": None})
    obj = store.get("x")
    assert obj.vectors["v"] is None
    # Property untouched
    assert obj.properties["title"] == "a"


def test_adds_new_property_column_on_update(store):
    store.add("x", properties={"title": "a"})
    store.update("x", properties={"new_field": 42})
    obj = store.get("x")
    assert obj.properties["new_field"] == 42


def test_update_preserves_other_rows(store):
    store.register_vector_field("v", dim=2)
    store.add("x1", properties={"title": "a"}, vectors={"v": [1.0, 0.0]})
    store.add("x2", properties={"title": "b"}, vectors={"v": [0.0, 1.0]})
    store.update("x1", properties={"title": "A"})

    assert store.get("x1").properties["title"] == "A"
    assert store.get("x2").properties["title"] == "b"
    assert store.get("x2").vectors["v"] == pytest.approx([0.0, 1.0])


def test_update_after_concurrent_delete_resurrects_partial_row(store, monkeypatch):
    # Regression test pinning the single-writer trade-off documented in
    # docs/architecture.md. update() checks exists() then issues a merge_insert
    # with when_not_matched_insert_all(); if a concurrent writer deletes the
    # row in between, the merge_insert silently inserts a partial row whose
    # untouched columns are null. We don't defend against this (single-writer
    # contract), but we pin the behavior so we notice if LanceDB changes it.
    store.add("x", properties={"title": "a", "tag": "keep"})

    backend = store._backend
    real_exists = backend.exists
    calls = {"n": 0}

    def racing_exists(object_id: str) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            assert real_exists(object_id)
            backend._table.delete(f"object_id = '{object_id}'")
            return True
        return real_exists(object_id)

    monkeypatch.setattr(backend, "exists", racing_exists)

    store.update("x", properties={"title": "B"})

    obj = store.get("x")
    assert obj is not None
    assert obj.properties["title"] == "B"
    # The column the update didn't touch is now null because the partial
    # row was inserted fresh, not merged into the (deleted) original.
    assert obj.properties.get("tag") is None
