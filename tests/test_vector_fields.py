from __future__ import annotations

import pytest

from object_vectordb import (
    DimensionMismatch,
    ObjectVectorDB,
    SchemaError,
    VectorFieldNotRegistered,
)


def test_register_returns_info(store):
    info = store.register_vector_field("clip", dim=512, description="CLIP ViT-B/32")
    assert info.name == "clip"
    assert info.dim == 512
    assert info.has_index is False
    assert info.description == "CLIP ViT-B/32"


def test_register_idempotent_on_same_dim(store):
    store.register_vector_field("v", dim=128)
    store.register_vector_field("v", dim=128)
    assert len(store.vector_fields()) == 1


def test_register_conflicting_dim_raises(store):
    store.register_vector_field("v", dim=128)
    with pytest.raises(DimensionMismatch):
        store.register_vector_field("v", dim=256)


def test_register_is_zero_copy(store):
    store.add("x", properties={"title": "a"})
    store.register_vector_field("v", dim=4)
    # Existing row is preserved; new vector field is None.
    obj = store.get("x")
    assert obj.properties["title"] == "a"
    assert obj.vectors["v"] is None


def test_write_without_register_raises(store):
    with pytest.raises(VectorFieldNotRegistered):
        store.add("x", vectors={"unregistered": [1.0, 2.0]})


def test_auto_register(tmp_path):
    s = ObjectVectorDB(uri=str(tmp_path / "db"), auto_register=True)
    s.add("x", vectors={"clip": [1.0, 2.0, 3.0]})
    fields = {v.name: v.dim for v in s.vector_fields()}
    assert fields == {"clip": 3}


def test_dim_mismatch_on_write_raises(store):
    store.register_vector_field("v", dim=4)
    with pytest.raises(DimensionMismatch):
        store.add("x", vectors={"v": [1.0, 2.0]})


def test_drop_vector_field(store):
    store.register_vector_field("v", dim=3)
    store.add("x", vectors={"v": [1.0, 2.0, 3.0]})

    store.drop_fields(["v"])
    names = {f.name for f in store.vector_fields()}
    assert "v" not in names
    # Object still exists, vectors dict no longer includes v
    obj = store.get("x")
    assert "v" not in obj.vectors


def test_drop_property_field(store):
    store.add("x", properties={"title": "a", "views": 10})
    store.drop_fields(["title"])
    schema = store.schema()
    assert "title" not in schema["properties"]
    assert "views" in schema["properties"]
    obj = store.get("x")
    assert "title" not in obj.properties
    assert obj.properties["views"] == 10


def test_drop_nonexistent_is_noop(store):
    store.drop_fields(["never_existed"])


def test_rename_property_field(store):
    store.add("x", properties={"title": "a"})
    store.rename_field("title", "caption")
    obj = store.get("x")
    assert "caption" in obj.properties
    assert obj.properties["caption"] == "a"
    assert "title" not in obj.properties


def test_rename_vector_field(store):
    store.register_vector_field("v", dim=2)
    store.add("x", vectors={"v": [1.0, 2.0]})
    store.rename_field("v", "w")
    fields = {f.name for f in store.vector_fields()}
    assert fields == {"w"}
    obj = store.get("x")
    assert obj.vectors["w"] == pytest.approx([1.0, 2.0])


def test_property_name_cannot_use_reserved_prefix(store):
    with pytest.raises(SchemaError):
        store.add("x", properties={"__vec_sneaky": 1})


def test_register_rejects_reserved_prefix(store):
    with pytest.raises(SchemaError):
        store.register_vector_field("__vec_foo", dim=4)
