from __future__ import annotations

import pytest

from object_store import ObjectStore


def test_close_and_reopen(tmp_path):
    uri = str(tmp_path / "db")

    s1 = ObjectStore(uri=uri, table_name="objs")
    s1.register_vector_field("v", dim=3, description="my vecs")
    s1.add("x", properties={"title": "hello"}, vectors={"v": [1.0, 2.0, 3.0]})
    del s1

    s2 = ObjectStore(uri=uri, table_name="objs")
    assert s2.exists("x")
    obj = s2.get("x")
    assert obj.properties["title"] == "hello"
    assert obj.vectors["v"] == pytest.approx([1.0, 2.0, 3.0])
    fields = {f.name: f for f in s2.vector_fields()}
    assert "v" in fields
    assert fields["v"].dim == 3
    assert fields["v"].description == "my vecs"


def test_index_survives_reopen(tmp_path):
    uri = str(tmp_path / "db")

    s1 = ObjectStore(uri=uri, table_name="objs")
    s1.register_vector_field("v", dim=4)
    for i in range(300):
        s1.add(f"id{i}", vectors={"v": [float(i), 0.0, 0.0, 0.0]})
    s1.create_index("v", index_type="IVF_PQ", metric="cosine", num_partitions=2, num_sub_vectors=1)
    del s1

    s2 = ObjectStore(uri=uri, table_name="objs")
    info = s2.index_info("v")
    assert info is not None
    assert info.index_type == "IVF_PQ"
    assert info.metric == "cosine"
    assert info.params.get("num_partitions") == 2


def test_schema_survives_reopen(tmp_path):
    uri = str(tmp_path / "db")

    s1 = ObjectStore(uri=uri, table_name="objs")
    s1.add("x", properties={"title": "a", "views": 10})
    del s1

    s2 = ObjectStore(uri=uri, table_name="objs")
    sc = s2.schema()
    assert "title" in sc["properties"]
    assert "views" in sc["properties"]
