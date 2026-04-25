"""Per-collection schema registry backed by Arrow field metadata.

Vector field records (name, dim, description, index config) are stored in the
metadata of each __vec_<name> Arrow field inside the Lance manifest. Property
columns are inferred from the table schema at read time.  The sentinel key on
the object_id field marks a table as an ovdb collection.

Multi-writer safe: all metadata writes go through replace_field_metadata /
add_columns, both of which update the Lance manifest.  LanceDB retries manifest
commit conflicts automatically, so concurrent schema mutations are safe.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

VECTOR_COLUMN_PREFIX = "__vec_"
_OBJECT_ID_COLUMN = "object_id"

# Sentinel stored in the object_id field metadata to identify ovdb collections.
SCHEMA_SENTINEL_KEY = "ovdb_schema_version"
SCHEMA_SENTINEL_VALUE = "1"

# Arrow field metadata keys for vector fields (string on write; bytes on read).
_META_DIM = "ovdb_dim"
_META_DESC = "ovdb_description"
_META_INDEX = "ovdb_index"

# Retry configuration for manifest-touching writes.
_RETRY_MAX_ATTEMPTS = 5
_RETRY_BASE_MS = 50

# Try to import Lance's native commit-conflict exception for precise matching.
try:
    from lancedb._lancedb import CommitConflictError as _LanceConflict  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    _LanceConflict = None  # type: ignore[assignment,misc]


def _is_conflict(exc: BaseException) -> bool:
    if _LanceConflict is not None and isinstance(exc, _LanceConflict):
        return True
    msg = str(exc).lower()
    return (
        "commit conflict" in msg
        or "conflicting transaction" in msg
        or "incompatible transaction" in msg
    )


def _with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying on Lance manifest commit conflicts."""
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_conflict(exc) or attempt == _RETRY_MAX_ATTEMPTS - 1:
                raise
            delay = _RETRY_BASE_MS * (2**attempt) / 1000
            log.debug("Commit conflict on attempt %d; retrying in %.3fs", attempt + 1, delay)
            time.sleep(delay)


@dataclass
class VectorFieldRecord:
    name: str
    dim: int
    column: str
    description: str | None = None
    index: dict[str, Any] | None = None


def _is_ovdb_table(table) -> bool:
    """Return True if the Lance table was created by object_vectordb."""
    try:
        field = table.schema.field(_OBJECT_ID_COLUMN)
        meta = field.metadata or {}
        return meta.get(SCHEMA_SENTINEL_KEY.encode()) == SCHEMA_SENTINEL_VALUE.encode()
    except (KeyError, Exception):
        return False


class SchemaRegistry:
    """Collection-level operations backed by table_names() and field-metadata sentinels.

    Holds no on-disk state: collection discovery uses db.table_names() filtered
    by the `ovdb_schema_version` sentinel on each table's object_id field.
    """

    def __init__(self, uri: str, db) -> None:
        self._uri = uri
        self._db = db

    def collection(self, name: str) -> CollectionRegistry:
        return CollectionRegistry()

    def list_collections(self) -> list[str]:
        result = []
        for name in self._db.table_names():
            try:
                if _is_ovdb_table(self._db.open_table(name)):
                    result.append(name)
            except Exception:
                pass
        return sorted(result)

    def has_collection(self, name: str) -> bool:
        if name not in self._db.table_names():
            return False
        try:
            return _is_ovdb_table(self._db.open_table(name))
        except Exception:
            return False

    def drop_collection(self, name: str) -> None:
        pass  # No registry state to clean up; table is dropped by db.py.


class CollectionRegistry:
    """Per-collection registry backed by Arrow field metadata on the Lance table.

    Call bind_table(table) right after the table is opened or created by the backend.
    Until bind_table is called no methods should be invoked.
    """

    def __init__(self) -> None:
        self._table = None

    def bind_table(self, table) -> None:
        self._table = table

    # ---- column name helper ----

    def vector_column(self, name: str) -> str:
        return VECTOR_COLUMN_PREFIX + name

    # ---- internal metadata helpers ----

    def _read_vec_meta(self, column: str) -> dict[str, Any] | None:
        """Return parsed metadata dict for a vector column, or None if not registered."""
        try:
            field = self._table.schema.field(column)
        except KeyError:
            return None
        meta = field.metadata or {}
        dim_b = meta.get(_META_DIM.encode())
        if dim_b is None:
            return None
        desc_b = meta.get(_META_DESC.encode(), b"")
        idx_b = meta.get(_META_INDEX.encode(), b"")
        return {
            "dim": int(dim_b),
            "description": desc_b.decode() or None,
            "index": json.loads(idx_b) if idx_b else None,
        }

    def _build_record(self, name: str, meta: dict[str, Any]) -> VectorFieldRecord:
        return VectorFieldRecord(
            name=name,
            dim=meta["dim"],
            column=self.vector_column(name),
            description=meta["description"],
            index=meta["index"],
        )

    def _flat_meta(
        self, dim: int, description: str | None, index: dict[str, Any] | None
    ) -> dict[str, str]:
        return {
            _META_DIM: str(dim),
            _META_DESC: description or "",
            _META_INDEX: json.dumps(index, sort_keys=True) if index else "",
        }

    # ---- public vector-field interface ----

    def has_vector(self, name: str) -> bool:
        return self._read_vec_meta(self.vector_column(name)) is not None

    def get_vector(self, name: str) -> VectorFieldRecord | None:
        meta = self._read_vec_meta(self.vector_column(name))
        if meta is None:
            return None
        return self._build_record(name, meta)

    def list_vectors(self) -> list[VectorFieldRecord]:
        recs = []
        for f in self._table.schema:
            if not f.name.startswith(VECTOR_COLUMN_PREFIX):
                continue
            meta = self._read_vec_meta(f.name)
            if meta is None:
                continue
            field_name = f.name[len(VECTOR_COLUMN_PREFIX) :]
            recs.append(self._build_record(field_name, meta))
        return recs

    def add_vector(self, name: str, dim: int, description: str | None = None) -> VectorFieldRecord:
        column = self.vector_column(name)
        _with_retry(
            self._table.replace_field_metadata, column, self._flat_meta(dim, description, None)
        )
        return VectorFieldRecord(name=name, dim=dim, column=column, description=description)

    def remove_vector(self, name: str) -> None:
        pass  # Column is dropped directly from the table by the backend.

    def rename_vector(self, old: str, new: str) -> VectorFieldRecord:
        # alter_columns rename preserves field metadata, so just return the record at
        # the new column name.  The backend has already called alter_columns before us.
        new_col = self.vector_column(new)
        meta = self._read_vec_meta(new_col)
        if meta is None:
            raise RuntimeError(f"Expected vector column {new_col!r} not found after rename.")
        return self._build_record(new, meta)

    def set_index(self, name: str, index: dict[str, Any] | None) -> None:
        rec = self.get_vector(name)
        if rec is None:
            return
        _with_retry(
            self._table.replace_field_metadata,
            rec.column,
            self._flat_meta(rec.dim, rec.description, index),
        )

    def update_description(self, name: str, description: str | None) -> None:
        rec = self.get_vector(name)
        if rec is None:
            return
        _with_retry(
            self._table.replace_field_metadata,
            rec.column,
            self._flat_meta(rec.dim, description, rec.index),
        )

    # ---- property columns — derived from schema, no registry writes needed ----

    def has_property(self, name: str) -> bool:
        names = {f.name for f in self._table.schema}
        return (
            name in names
            and not name.startswith(VECTOR_COLUMN_PREFIX)
            and name != _OBJECT_ID_COLUMN
        )

    def add_property(self, name: str) -> None:
        pass

    def remove_property(self, name: str) -> None:
        pass

    def rename_property(self, old: str, new: str) -> None:
        pass

    def list_properties(self) -> list[str]:
        return sorted(
            f.name
            for f in self._table.schema
            if f.name != _OBJECT_ID_COLUMN and not f.name.startswith(VECTOR_COLUMN_PREFIX)
        )
