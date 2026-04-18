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


def test_add_without_vectors_allowed(store):
    store.add("x", properties={"title": "no vec"})
    obj = store.get("x")
    assert obj.properties["title"] == "no vec"
    assert obj.vectors == {}


def test_add_without_properties_allowed(store):
    store.register_vector_field("v", dim=2)
    store.add("x", vectors={"v": [0.0, 1.0]})
    assert store.get("x").vectors["v"] == pytest.approx([0.0, 1.0])
