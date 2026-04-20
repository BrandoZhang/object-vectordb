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


def _racing_delete_exists(backend, object_id_to_delete: str):
    """Build an `exists` replacement that, on its first call, deletes the row
    via the backend's underlying table and still returns True. Simulates a
    concurrent writer winning a delete between the pre-check and the merge."""
    real_exists = backend.exists
    calls = {"n": 0}

    def racing_exists(object_id: str) -> bool:
        calls["n"] += 1
        if calls["n"] == 1 and object_id == object_id_to_delete:
            assert real_exists(object_id)
            backend._table.delete(f"object_id = '{object_id}'")
            return True
        return real_exists(object_id)

    return racing_exists


def test_update_default_silently_noops_when_concurrently_deleted(store, monkeypatch):
    # With the default on_missing="raise", the merge_insert is update-only
    # (no when_not_matched_insert_all), so a row deleted between the
    # exists() pre-check and the merge_insert stays deleted — strictly
    # better than the partial-row resurrection we'd get from upsert.
    store.add("x", properties={"title": "a", "tag": "keep"})
    backend = store._backend
    monkeypatch.setattr(backend, "exists", _racing_delete_exists(backend, "x"))

    store.update("x", properties={"title": "B"})

    assert store.get("x") is None


def test_update_on_missing_insert_after_delete_writes_partial_row(store):
    # Pin the upsert behavior: under on_missing="insert" the merge runs as
    # an upsert (when_not_matched_insert_all). If the row was deleted
    # before the merge ran, the new row contains ONLY the columns the
    # update touched — prior column values do not carry over.
    store.add("x", properties={"title": "a", "tag": "keep"})
    store.delete("x")

    store.update("x", properties={"title": "B"}, on_missing="insert")

    obj = store.get("x")
    assert obj is not None
    assert obj.properties["title"] == "B"
    assert obj.properties.get("tag") is None


def test_update_on_missing_insert_creates_row_when_absent(store):
    store.update("new-id", properties={"title": "fresh"}, on_missing="insert")
    obj = store.get("new-id")
    assert obj is not None
    assert obj.properties["title"] == "fresh"


def test_update_on_missing_skip_is_silent_noop(store):
    store.update("ghost", properties={"title": "x"}, on_missing="skip")
    assert store.get("ghost") is None


def test_update_on_missing_invalid_raises(store):
    store.add("x", properties={"title": "a"})
    with pytest.raises(ValueError):
        store.update("x", properties={"title": "b"}, on_missing="bogus")
