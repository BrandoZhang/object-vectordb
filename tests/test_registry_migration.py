"""Tests for the one-time migration from old JSON sidecar to Arrow field metadata."""

from __future__ import annotations

import json
from pathlib import Path

from object_vectordb import ObjectVectorDB
from object_vectordb.registry import (
    REGISTRY_FILENAME,
    _is_ovdb_table,
)


def _build_old_sidecar(uri: str, collections: dict) -> None:
    """Write a v2 JSON sidecar as the old SDK would have."""
    sidecar = Path(uri) / REGISTRY_FILENAME
    state = {
        "version": 2,
        "collections": collections,
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w") as f:
        json.dump(state, f, indent=2)


def _build_old_collection(uri: str, name: str, vector_fields: dict, properties: list) -> None:
    """Create the Lance table as the old SDK would have, without Arrow metadata."""
    import lancedb
    import pyarrow as pa

    from object_vectordb.registry import VECTOR_COLUMN_PREFIX

    db = lancedb.connect(uri)
    schema = pa.schema([pa.field("object_id", pa.string(), nullable=False)])
    table = db.create_table(name, schema=schema)
    for vf_name, vf_info in vector_fields.items():
        dim = vf_info["dim"]
        col = VECTOR_COLUMN_PREFIX + vf_name
        table.add_columns(pa.field(col, pa.list_(pa.float32(), dim)))
    for prop in properties:
        table.add_columns(pa.field(prop, pa.string()))


# ---------------------------------------------------------------------------
# Migration: sidecar is read, metadata is written, sidecar is deleted
# ---------------------------------------------------------------------------


def test_migration_writes_field_metadata_and_removes_sidecar(tmp_path):
    uri = str(tmp_path / "db")
    coll_name = "items"
    vf = {
        "embed": {
            "name": "embed",
            "dim": 8,
            "column": "__vec_embed",
            "description": "desc",
            "index": None,
        }
    }
    _build_old_collection(uri, coll_name, {"embed": {"dim": 8}}, ["title"])
    _build_old_sidecar(uri, {coll_name: {"vector_fields": vf, "property_columns": ["title"]}})

    # Sidecar should exist before migration.
    assert (tmp_path / "db" / REGISTRY_FILENAME).exists()

    # Opening the DB triggers migration.
    db = ObjectVectorDB(uri=uri)
    col = db.collection(coll_name)

    # Sidecar must be gone.
    assert not (tmp_path / "db" / REGISTRY_FILENAME).exists()

    # Vector field metadata must be present.
    recs = col.list_vector_fields()
    assert len(recs) == 1
    assert recs[0].name == "embed"
    assert recs[0].dim == 8


def test_migration_sentinel_written_on_object_id(tmp_path):
    uri = str(tmp_path / "db")
    coll_name = "things"
    vf = {"v": {"name": "v", "dim": 4, "column": "__vec_v", "description": None, "index": None}}
    _build_old_collection(uri, coll_name, {"v": {"dim": 4}}, [])
    _build_old_sidecar(uri, {coll_name: {"vector_fields": vf, "property_columns": []}})

    ObjectVectorDB(uri=uri)  # trigger migration

    import lancedb

    db = lancedb.connect(uri)
    table = db.open_table(coll_name)
    assert _is_ovdb_table(table)


def test_migration_index_config_preserved(tmp_path):
    uri = str(tmp_path / "db")
    coll_name = "indexed"
    index_cfg = {"type": "IVF_PQ", "metric": "cosine"}
    vf = {
        "clip": {
            "name": "clip",
            "dim": 16,
            "column": "__vec_clip",
            "description": None,
            "index": index_cfg,
        }
    }
    _build_old_collection(uri, coll_name, {"clip": {"dim": 16}}, [])
    _build_old_sidecar(uri, {coll_name: {"vector_fields": vf, "property_columns": []}})

    ObjectVectorDB(uri=uri)

    col = ObjectVectorDB(uri=uri).collection(coll_name)
    info = col.index_info("clip")
    assert info is not None
    assert info.metric == "cosine"
    assert info.index_type == "IVF_PQ"


def test_migration_is_idempotent(tmp_path):
    """Opening the DB a second time after migration does not fail or re-write."""
    uri = str(tmp_path / "db")
    coll_name = "idempotent"
    vf = {"v": {"name": "v", "dim": 4, "column": "__vec_v", "description": None, "index": None}}
    _build_old_collection(uri, coll_name, {"v": {"dim": 4}}, [])
    _build_old_sidecar(uri, {coll_name: {"vector_fields": vf, "property_columns": []}})

    ObjectVectorDB(uri=uri)  # first open — migrates
    assert not (tmp_path / "db" / REGISTRY_FILENAME).exists()

    # Second open — no sidecar, should be fine.
    db2 = ObjectVectorDB(uri=uri)
    col = db2.collection(coll_name)
    recs = col.list_vector_fields()
    assert recs[0].dim == 4


# ---------------------------------------------------------------------------
# New collections (no sidecar) get sentinel automatically
# ---------------------------------------------------------------------------


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
