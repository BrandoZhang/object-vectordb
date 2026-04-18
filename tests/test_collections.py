from __future__ import annotations

import pytest

from object_vectordb import (
    DimensionMismatch,
    ObjectVectorDB,
    VectorFieldNotRegistered,
)


def test_collection_creates_on_first_access(db):
    assert db.list_collections() == []
    db.collection("videos")
    assert db.list_collections() == ["videos"]


def test_list_and_has_collection(db):
    db.collection("a")
    db.collection("b")
    db.collection("c")
    assert db.list_collections() == ["a", "b", "c"]
    assert db.has_collection("a")
    assert not db.has_collection("missing")


def test_drop_collection(db):
    c = db.collection("to_drop")
    c.register_vector_field("v", dim=3)
    c.add("x", vectors={"v": [1.0, 2.0, 3.0]})

    db.drop_collection("to_drop")

    assert not db.has_collection("to_drop")
    assert db.list_collections() == []


def test_drop_missing_collection_is_noop(db):
    db.drop_collection("never_existed")


def test_vector_fields_isolated_across_collections(db):
    videos = db.collection("videos")
    images = db.collection("images")

    videos.register_vector_field("clip", dim=3)
    images.register_vector_field("siglip", dim=4)

    videos_fields = {f.name for f in videos.vector_fields()}
    images_fields = {f.name for f in images.vector_fields()}

    assert videos_fields == {"clip"}
    assert images_fields == {"siglip"}


def test_same_field_name_different_dim_per_collection(db):
    """Two collections can both register a field named 'v' with different dims."""
    a = db.collection("a")
    b = db.collection("b")
    a.register_vector_field("v", dim=3)
    b.register_vector_field("v", dim=5)

    a.add("x", vectors={"v": [1.0, 2.0, 3.0]})
    b.add("x", vectors={"v": [1.0, 2.0, 3.0, 4.0, 5.0]})

    # Dim enforcement is per-collection
    with pytest.raises(DimensionMismatch):
        a.add("y", vectors={"v": [1.0, 2.0, 3.0, 4.0, 5.0]})


def test_search_cannot_cross_collection_boundary(db):
    videos = db.collection("videos")
    images = db.collection("images")

    videos.register_vector_field("clip", dim=3)
    videos.add("v1", vectors={"clip": [1.0, 0.0, 0.0]})

    with pytest.raises(VectorFieldNotRegistered):
        images.search([1.0, 0.0, 0.0], vector_field="clip", limit=1)


def test_reopen_roundtrips_collections(tmp_path):
    uri = str(tmp_path / "db")

    db1 = ObjectVectorDB(uri=uri)
    a = db1.collection("a")
    b = db1.collection("b")
    a.register_vector_field("v", dim=3)
    b.register_vector_field("w", dim=4)
    a.add("x", vectors={"v": [1.0, 2.0, 3.0]})
    del a, b, db1

    db2 = ObjectVectorDB(uri=uri)
    assert set(db2.list_collections()) == {"a", "b"}
    a2 = db2.collection("a")
    b2 = db2.collection("b")
    assert [f.name for f in a2.vector_fields()] == ["v"]
    assert [f.name for f in b2.vector_fields()] == ["w"]
    assert a2.get("x").vectors["v"] == pytest.approx([1.0, 2.0, 3.0])


def test_empty_collection_name_rejected(db):
    with pytest.raises(ValueError):
        db.collection("")
