from __future__ import annotations

import pytest

from object_store import ObjectNotFound


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
