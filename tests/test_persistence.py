from __future__ import annotations

import pytest

from object_vectordb import ObjectVectorDB


def test_close_and_reopen(tmp_path):
    uri = str(tmp_path / "db")

    db1 = ObjectVectorDB(uri=uri)
    c1 = db1.collection("objs")
    c1.register_vector_field("v", dim=3, description="my vecs")
    c1.add("x", properties={"title": "hello"}, vectors={"v": [1.0, 2.0, 3.0]})
    del c1, db1

    db2 = ObjectVectorDB(uri=uri)
    c2 = db2.collection("objs")
    assert c2.exists("x")
    obj = c2.get("x")
    assert obj.properties["title"] == "hello"
    assert obj.vectors["v"] == pytest.approx([1.0, 2.0, 3.0])
    fields = {f.name: f for f in c2.list_vector_fields()}
    assert "v" in fields
    assert fields["v"].dim == 3
    assert fields["v"].description == "my vecs"


def test_index_survives_reopen(tmp_path):
    uri = str(tmp_path / "db")

    db1 = ObjectVectorDB(uri=uri)
    c1 = db1.collection("objs")
    c1.register_vector_field("v", dim=4)
    for i in range(300):
        c1.add(f"id{i}", vectors={"v": [float(i), 0.0, 0.0, 0.0]})
    c1.create_index("v", index_type="IVF_PQ", metric="cosine", num_partitions=2, num_sub_vectors=1)
    del c1, db1

    db2 = ObjectVectorDB(uri=uri)
    c2 = db2.collection("objs")
    info = c2.index_info("v")
    assert info is not None
    assert info.index_type == "IVF_PQ"
    assert info.metric == "cosine"
    assert info.params.get("num_partitions") == 2


def test_schema_survives_reopen(tmp_path):
    uri = str(tmp_path / "db")

    db1 = ObjectVectorDB(uri=uri)
    c1 = db1.collection("objs")
    c1.add("x", properties={"title": "a", "views": 10})
    del c1, db1

    db2 = ObjectVectorDB(uri=uri)
    c2 = db2.collection("objs")
    sc = c2.schema()
    assert "title" in sc["properties"]
    assert "views" in sc["properties"]
