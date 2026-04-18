from __future__ import annotations

import pytest

from object_vectordb import SchemaError


def test_schema_reports_properties_and_vectors_separately(store):
    store.register_vector_field("clip", dim=512)
    store.add("x", properties={"title": "a", "views": 10, "tags": ["a", "b"]})
    schema = store.schema()
    assert "title" in schema["properties"]
    assert "views" in schema["properties"]
    assert "tags" in schema["properties"]
    assert "clip" in schema["vectors"]
    assert schema["vectors"]["clip"]["dim"] == 512


def test_auto_add_property_int(store):
    store.add("x", properties={"a": 1})
    assert "a" in store.schema()["properties"]


def test_auto_add_property_float(store):
    store.add("x", properties={"f": 3.14})
    assert "f" in store.schema()["properties"]


def test_auto_add_property_bool(store):
    store.add("x", properties={"flag": True})
    obj = store.get("x")
    assert obj.properties["flag"] is True


def test_auto_add_property_str(store):
    store.add("x", properties={"s": "hello"})
    obj = store.get("x")
    assert obj.properties["s"] == "hello"


def test_auto_add_property_list(store):
    store.add("x", properties={"items": ["a", "b", "c"]})
    obj = store.get("x")
    assert obj.properties["items"] == ["a", "b", "c"]


def test_auto_add_property_bytes(store):
    store.add("x", properties={"blob": b"\x00\x01\x02"})
    obj = store.get("x")
    assert obj.properties["blob"] == b"\x00\x01\x02"


def test_auto_add_property_none_rejected(store):
    with pytest.raises(SchemaError):
        store.add("x", properties={"mystery": None})


def test_property_survives_schema_grow(store):
    store.add("a", properties={"title": "a"})
    store.add("b", properties={"title": "b", "views": 10})
    # 'a' has no views column assignment but it exists now
    obj_a = store.get("a")
    assert obj_a.properties["title"] == "a"
    assert obj_a.properties["views"] is None
    obj_b = store.get("b")
    assert obj_b.properties["views"] == 10
