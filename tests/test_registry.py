"""Tests for the sentinel-based collection discovery.

Vector field records live in Arrow field metadata on each Lance manifest.
An `ovdb_schema_version` sentinel on the object_id field marks a table as
an ovdb collection so `list_collections()` can distinguish ours from
foreign Lance tables at the same URI.
"""

from __future__ import annotations

from object_vectordb import ObjectVectorDB
from object_vectordb.registry import _is_ovdb_table


def test_new_collection_has_sentinel(tmp_path):
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    db.collection("fresh")

    import lancedb

    raw_db = lancedb.connect(str(tmp_path / "db"))
    table = raw_db.open_table("fresh")
    assert _is_ovdb_table(table)


def test_list_collections_excludes_non_ovdb_tables(tmp_path):
    import lancedb
    import pyarrow as pa

    uri = str(tmp_path / "db")
    # Create an ovdb collection.
    db = ObjectVectorDB(uri=uri)
    db.collection("mine")

    # Also create a raw LanceDB table without the sentinel.
    raw = lancedb.connect(uri)
    raw.create_table("foreign", schema=pa.schema([pa.field("x", pa.int64())]))

    assert db.list_collections() == ["mine"]
