from __future__ import annotations

import pytest

from object_vectordb import DuplicateObject, ObjectData


def test_add_get_delete_roundtrip(store):
    store.register_vector_field("v", dim=3)
    store.add("x1", properties={"title": "hello"}, vectors={"v": [1.0, 0.0, 0.0]})

    obj = store.get("x1")
    assert isinstance(obj, ObjectData)
    assert obj.object_id == "x1"
    assert obj.properties["title"] == "hello"
    assert obj.vectors["v"] == pytest.approx([1.0, 0.0, 0.0])

    assert store.exists("x1") is True
    store.delete("x1")
    assert store.exists("x1") is False
    assert store.get("x1") is None


def test_get_missing_returns_none(store):
    assert store.get("does-not-exist") is None


def test_exists_missing_returns_false(store):
    assert store.exists("nope") is False


def test_add_duplicate_raises(store):
    store.register_vector_field("v", dim=2)
    store.add("x", properties={}, vectors={"v": [1.0, 0.0]})
    with pytest.raises(DuplicateObject):
        store.add("x", properties={}, vectors={"v": [0.0, 1.0]})


def test_delete_missing_is_silent_noop(store):
    store.delete("does-not-exist")
    store.delete("does-not-exist")


def test_add_after_delete_reuses_id(store):
    store.register_vector_field("v", dim=2)
    store.add("x", properties={"title": "first", "tag": "old"}, vectors={"v": [1.0, 0.0]})
    store.delete("x")
    store.add("x", properties={"title": "second"}, vectors={"v": [0.0, 1.0]})

    obj = store.get("x")
    assert obj.properties["title"] == "second"
    # The re-added row should not carry over properties from the deleted one.
    assert obj.properties.get("tag") is None
    assert obj.vectors["v"] == pytest.approx([0.0, 1.0])


def test_add_without_vectors_allowed(store):
    store.add("x", properties={"title": "no vec"})
    obj = store.get("x")
    assert obj.properties["title"] == "no vec"
    assert obj.vectors == {}


def test_add_without_properties_allowed(store):
    store.register_vector_field("v", dim=2)
    store.add("x", vectors={"v": [0.0, 1.0]})
    assert store.get("x").vectors["v"] == pytest.approx([0.0, 1.0])


# ---------------------------------------------------------------------------
# batch_get
# ---------------------------------------------------------------------------


def test_batch_get_returns_objects_in_input_order(store):
    store.register_vector_field("v", dim=2)
    for oid, n in [("a", 1), ("b", 2), ("c", 3)]:
        store.add(oid, properties={"n": n}, vectors={"v": [float(n), 0.0]})
    hits = store.batch_get(["c", "a", "b"])
    assert [h.object_id for h in hits] == ["c", "a", "b"]
    assert [h.properties["n"] for h in hits] == [3, 1, 2]


def test_batch_get_missing_ids_yield_none(store):
    store.add("a", properties={"n": 1})
    hits = store.batch_get(["a", "ghost", "a"])
    assert hits[0].object_id == "a"
    assert hits[1] is None
    assert hits[2].object_id == "a"


def test_batch_get_empty_input_returns_empty_list(store):
    store.add("a", properties={"n": 1})
    assert store.batch_get([]) == []


def test_batch_get_all_missing_returns_all_none(store):
    assert store.batch_get(["x", "y"]) == [None, None]


def test_batch_get_accepts_iterable(store):
    store.add("a", properties={"n": 1})
    store.add("b", properties={"n": 2})
    hits = store.batch_get(iter(["a", "b"]))
    assert [h.object_id for h in hits] == ["a", "b"]


# ---------------------------------------------------------------------------
# Scalar index on object_id (auto-created)
# ---------------------------------------------------------------------------


def test_object_id_scalar_index_is_auto_created(store):
    # The index exists immediately after the Collection is constructed, even
    # before any rows are added — we create it on table bootstrap so lookups
    # are O(log N) from the first write onwards.
    indices = store._backend._table.list_indices()
    assert any(getattr(i, "columns", None) == ["object_id"] for i in indices)


def test_object_id_scalar_index_survives_reopen(tmp_path):
    from object_vectordb import ObjectVectorDB

    uri = str(tmp_path / "db")
    col = ObjectVectorDB(uri=uri).collection("c")
    col.add("a", properties={"n": 1})
    # Reopen — the index should already exist (and _ensure_object_id_index
    # should no-op rather than duplicate).
    col2 = ObjectVectorDB(uri=uri).collection("c")
    indices = col2._backend._table.list_indices()
    matches = [i for i in indices if getattr(i, "columns", None) == ["object_id"]]
    assert len(matches) == 1
