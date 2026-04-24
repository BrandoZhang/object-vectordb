"""LanceDB backend. All lancedb and pyarrow code lives here.

The public `ObjectVectorDB` class in `db.py` delegates to this backend through
Python-native types only. Swapping backends (e.g. to Qdrant/Milvus) would mean
writing a new class with the same method signatures and replacing the import
in `db.py`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pyarrow as pa

from .arrow_utils import (
    arrow_type_to_sql_type,
    encode_property_value,
    python_value_to_arrow_type,
)
from .exceptions import (
    DimensionMismatch,
    DuplicateObject,
    MetricMismatch,
    ObjectNotFound,
    SchemaError,
    VectorFieldNotRegistered,
)
from .registry import (
    SCHEMA_SENTINEL_KEY,
    SCHEMA_SENTINEL_VALUE,
    VECTOR_COLUMN_PREFIX,
    CollectionRegistry,
    VectorFieldRecord,
    _with_retry,
)
from .scoring import distance_to_score, normalize_metric
from .types import IndexInfo, SearchResult, VectorFieldInfo

log = logging.getLogger(__name__)

OBJECT_ID_COLUMN = "object_id"
DEFAULT_METRIC = "cosine"
_UNBOUNDED_LIMIT = 2**31 - 1

# Probe for Lance's typed "already exists" exception so we can prefer
# isinstance-matching over fragile message-string matching.  The class name has
# moved between Lance releases, so we try several and fall back to string match.
_LanceAlreadyExists: tuple[type[BaseException], ...] = ()
for _candidate in ("AlreadyExistsError", "TableAlreadyExistsError"):
    try:
        _cls = getattr(__import__("lancedb._lancedb", fromlist=[_candidate]), _candidate)
        _LanceAlreadyExists = (*_LanceAlreadyExists, _cls)
    except (ImportError, AttributeError):
        pass


def _is_already_exists(exc: BaseException) -> bool:
    """Return True if exc signals that the target already exists.

    Covers two distinct Lance error surfaces that both mean "a concurrent
    writer got there first":

      * "already exists" — raised when the target (table / column / index)
        exists with a definition compatible with ours.
      * "type conflicts" — raised by Lance on a retried add_columns whose
        column was committed first by a concurrent writer with a different
        Arrow type or FixedSizeList dim.  We treat this the same as "already
        exists" so that _verify_property_column_type / _verify_vector_column_dim
        can produce a clean SchemaError / DimensionMismatch instead of letting
        a raw RuntimeError leak to the caller.

    Consolidates the string-match logic used across _ensure_table,
    _ensure_property_column, _ensure_vector_column, register_vector_field, and
    _ensure_object_id_index.  If LanceDB exposes typed exceptions for these in
    the future, adding them to _LanceAlreadyExists above is the only change.
    """
    if _LanceAlreadyExists and isinstance(exc, _LanceAlreadyExists):
        return True
    msg = str(exc).lower()
    return "already exists" in msg or "type conflicts" in msg


def _quote_literal(value: str) -> str:
    """Quote a string literal for a DataFusion SQL expression."""
    return "'" + value.replace("'", "''") + "'"


class LanceDBBackend:
    """All LanceDB-specific operations. One backend instance per Collection."""

    def __init__(
        self,
        db,
        table_name: str,
        registry: CollectionRegistry,
        auto_register: bool = False,
    ):
        self._db = db
        self._table_name = table_name
        self._registry = registry
        self._auto_register = auto_register
        self._table = self._ensure_table()
        self._registry.bind_table(self._table)

    # ------------------------------------------------------------------
    # Table bootstrap
    # ------------------------------------------------------------------

    def _ensure_table(self):
        created = False
        if self._table_name in self._db.table_names():
            table = self._db.open_table(self._table_name)
        else:
            sentinel_meta = {SCHEMA_SENTINEL_KEY.encode(): SCHEMA_SENTINEL_VALUE.encode()}
            schema = pa.schema(
                [pa.field(OBJECT_ID_COLUMN, pa.string(), nullable=False, metadata=sentinel_meta)]
            )
            try:
                table = self._db.create_table(self._table_name, schema=schema)
                created = True
            except (OSError, RuntimeError, ValueError) as exc:
                # Concurrent writer created the table first — open it instead.
                if not _is_already_exists(exc):
                    raise
                table = self._db.open_table(self._table_name)
        # Index creation is mandatory on new tables (we hold write access) and
        # best-effort on existing ones (a reader may have no write permission,
        # or a concurrent writer may have just created the index).
        self._ensure_object_id_index(table, required=created)
        return table

    @staticmethod
    def _ensure_object_id_index(table, required: bool = False) -> None:
        # BTREE on object_id turns every per-row existence check and id lookup
        # from O(N) scans into O(log N).
        if LanceDBBackend._has_object_id_index(table):
            return
        try:
            _with_retry(table.create_scalar_index, OBJECT_ID_COLUMN, index_type="BTREE")
            return
        except Exception as exc:
            # A concurrent writer just created the same index — success.
            if _is_already_exists(exc):
                return
            # Retry-exhausted "Incompatible transaction: CreateIndex ...
            # concurrent" or a transient failure.  If the index now exists
            # (because the concurrent writer's CreateIndex won the commit
            # race), treat it as success regardless of `required`: the end
            # state matches our intent.  Pass refresh=True because our table
            # handle's manifest view is pinned to the version we saw before
            # the race; list_indices() against the stale view will miss the
            # winner's just-committed CreateIndex transaction.
            if LanceDBBackend._has_object_id_index(table, refresh=True):
                return
            # Other failures (read-only credentials, unrelated errors) are
            # only fatal when we just created the table and genuinely need
            # the index.
            if required:
                raise
            log.debug("Could not create object_id BTREE index (non-fatal): %s", exc)

    @staticmethod
    def _has_object_id_index(table, refresh: bool = False) -> bool:
        # LanceDB Table handles pin a manifest version at open/create time;
        # list_indices() reads from that pinned view and will not see indices
        # committed by a concurrent writer on the same physical dataset.  When
        # we're checking *after* a concurrent race, refresh the view first.
        if refresh:
            try:
                table.checkout_latest()
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("checkout_latest failed: %s", exc)
        for idx in table.list_indices():
            if getattr(idx, "columns", None) == [OBJECT_ID_COLUMN]:
                return True
        return False

    def _table_columns(self) -> set[str]:
        return set(self._table.schema.names)

    def _column_type(self, name: str) -> pa.DataType:
        return self._table.schema.field(name).type

    # ------------------------------------------------------------------
    # Column ensurers (zero-copy metadata-only add_columns)
    # ------------------------------------------------------------------

    def _ensure_property_column(self, name: str, sample: Any) -> None:
        if name == OBJECT_ID_COLUMN:
            return
        if name.startswith(VECTOR_COLUMN_PREFIX):
            raise SchemaError(
                f"Property name {name!r} uses reserved prefix {VECTOR_COLUMN_PREFIX!r}."
            )
        if name in self._table_columns():
            return
        dtype = python_value_to_arrow_type(sample)
        try:
            _with_retry(self._table.add_columns, pa.field(name, dtype))
        except RuntimeError as exc:
            if not _is_already_exists(exc):
                raise
            # Concurrent writer added this property first.  Verify the winner's
            # inferred Arrow type matches ours; otherwise the caller would
            # silently coerce the value into the winner's type at encode time.
            self._verify_property_column_type(name, dtype)

    def _verify_property_column_type(self, name: str, inferred_dtype: pa.DataType) -> None:
        """Assert the existing property column has the type we'd have inferred.

        Called after a "column already exists" race — a concurrent writer may
        have added the column with a different inferred type (e.g. int64 from
        an int sample vs. string from ours), which would otherwise silently
        coerce our value into the winner's type at merge_insert time.
        """
        try:
            actual_type = self._table.schema.field(name).type
        except KeyError:
            raise SchemaError(f"Expected column {name!r} to exist after add_columns.") from None
        if actual_type != inferred_dtype:
            raise SchemaError(
                f"Property column {name!r} already exists with type {actual_type}; "
                f"cannot write a value that would have been inferred as {inferred_dtype}."
            )

    def _ensure_vector_column(self, name: str, dim: int) -> VectorFieldRecord:
        column = self._registry.vector_column(name)
        existing = self._registry.get_vector(name)
        if existing is not None:
            if existing.dim != dim:
                raise DimensionMismatch(name, existing.dim, dim)
            return existing
        if column not in self._table_columns():
            try:
                _with_retry(self._table.add_columns, pa.field(column, pa.list_(pa.float32(), dim)))
            except RuntimeError as exc:
                if not _is_already_exists(exc):
                    raise
        self._verify_vector_column_dim(name, column, dim)
        return self._registry.add_vector(name, dim)

    def _verify_vector_column_dim(self, name: str, column: str, dim: int) -> None:
        """Assert the existing column is a FixedSizeList with the requested dim.

        Called after a "column already exists" race — the concurrent winner
        may have used a different dim than we asked for, which would otherwise
        silently corrupt the metadata/column agreement.
        """
        try:
            actual_type = self._table.schema.field(column).type
        except KeyError:
            raise SchemaError(f"Expected column {column!r} to exist after add_columns.") from None
        if not isinstance(actual_type, pa.FixedSizeListType):
            raise SchemaError(f"Column {column!r} exists but is not a vector column.")
        if actual_type.list_size != dim:
            raise DimensionMismatch(name, actual_type.list_size, dim)

    # ------------------------------------------------------------------
    # Vector field registration
    # ------------------------------------------------------------------

    def register_vector_field(
        self, name: str, dim: int, description: str | None = None
    ) -> VectorFieldInfo:
        if not name:
            raise SchemaError("Vector field name must be a non-empty string.")
        if name.startswith(VECTOR_COLUMN_PREFIX):
            raise SchemaError(
                f"Vector field name {name!r} must not start with {VECTOR_COLUMN_PREFIX!r}."
            )
        existing = self._registry.get_vector(name)
        if existing is not None:
            if existing.dim != dim:
                raise DimensionMismatch(name, existing.dim, dim)
            if description is not None and existing.description != description:
                self._registry.update_description(name, description)
                existing = self._registry.get_vector(name)
            return self._vector_field_info(existing)
        column = self._registry.vector_column(name)
        if column not in self._table_columns():
            try:
                _with_retry(self._table.add_columns, pa.field(column, pa.list_(pa.float32(), dim)))
            except RuntimeError as exc:
                if not _is_already_exists(exc):
                    raise
        self._verify_vector_column_dim(name, column, dim)
        rec = self._registry.add_vector(name, dim, description)
        return self._vector_field_info(rec)

    def list_vector_fields(self) -> list[VectorFieldInfo]:
        return [self._vector_field_info(r) for r in self._registry.list_vectors()]

    def _vector_field_info(self, rec: VectorFieldRecord) -> VectorFieldInfo:
        has_index = bool(rec.index)
        return VectorFieldInfo(
            name=rec.name,
            dim=rec.dim,
            has_index=has_index,
            description=rec.description,
        )

    # ------------------------------------------------------------------
    # Write validation helpers
    # ------------------------------------------------------------------

    def _validate_vector(self, name: str, vector: list[float] | None) -> VectorFieldRecord:
        rec = self._registry.get_vector(name)
        if rec is None:
            if self._auto_register and vector is not None:
                rec = self._ensure_vector_column(name, len(vector))
            else:
                raise VectorFieldNotRegistered(name)
        if vector is not None and len(vector) != rec.dim:
            raise DimensionMismatch(name, rec.dim, len(vector))
        return rec

    def _prepare_row(
        self,
        object_id: str,
        properties: dict[str, Any] | None,
        vectors: dict[str, list[float] | None] | None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {OBJECT_ID_COLUMN: object_id}
        if properties:
            for key, value in properties.items():
                if value is None:
                    if key not in self._table_columns():
                        raise SchemaError(
                            f"Cannot add new property {key!r} with value=None "
                            f"(no type to infer). Write a non-None value first."
                        )
                    row[key] = None
                else:
                    self._ensure_property_column(key, value)
                    row[key] = encode_property_value(value, self._column_type(key))
        if vectors:
            for name, vec in vectors.items():
                rec = self._validate_vector(name, vec)
                row[rec.column] = vec
        return row

    def _batch_from_rows(self, rows: list[dict[str, Any]]) -> pa.Table:
        columns: dict[str, None] = {}
        for row in rows:
            for k in row:
                columns.setdefault(k, None)
        ordered = list(columns)
        table_schema = self._table.schema
        fields = [table_schema.field(name) for name in ordered]
        arrow_schema = pa.schema(fields)
        arrays = []
        for name, f in zip(ordered, fields, strict=True):
            data = [row.get(name) for row in rows]
            arrays.append(pa.array(data, type=f.type))
        return pa.Table.from_arrays(arrays, schema=arrow_schema)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def exists(self, object_id: str) -> bool:
        return self._table.count_rows(f"{OBJECT_ID_COLUMN} = {_quote_literal(object_id)}") > 0

    def _find_missing_ids(self, ids: list[str]) -> list[str]:
        """Return the subset of ids not present in the table (single scan)."""
        in_list = ", ".join(_quote_literal(i) for i in ids)
        rows = (
            self._table.search()
            .where(f"{OBJECT_ID_COLUMN} IN ({in_list})", prefilter=True)
            .limit(len(ids))
            .select([OBJECT_ID_COLUMN])
            .to_list()
        )
        found = {r[OBJECT_ID_COLUMN] for r in rows}
        return [i for i in ids if i not in found]

    def add(
        self,
        object_id: str,
        properties: dict[str, Any] | None,
        vectors: dict[str, list[float] | None] | None,
    ) -> None:
        row = self._prepare_row(object_id, properties, vectors)
        table = self._batch_from_rows([row])
        result = (
            self._table.merge_insert(OBJECT_ID_COLUMN).when_not_matched_insert_all().execute(table)
        )
        if result.num_inserted_rows == 0:
            raise DuplicateObject(object_id)

    def batch_add(
        self,
        items: list[dict[str, Any]],
    ) -> None:
        if not items:
            return
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            object_id = item["object_id"]
            if object_id in seen:
                raise DuplicateObject(object_id)
            seen.add(object_id)
            rows.append(self._prepare_row(object_id, item.get("properties"), item.get("vectors")))
        table = self._batch_from_rows(rows)
        result = (
            self._table.merge_insert(OBJECT_ID_COLUMN).when_not_matched_insert_all().execute(table)
        )
        if result.num_inserted_rows < len(rows):
            # Some ids were already present; find which ones (error path only).
            in_list = ", ".join(_quote_literal(i) for i in seen)
            found_rows = (
                self._table.search()
                .where(f"{OBJECT_ID_COLUMN} IN ({in_list})", prefilter=True)
                .limit(len(seen))
                .select([OBJECT_ID_COLUMN])
                .to_list()
            )
            found_ids = {r[OBJECT_ID_COLUMN] for r in found_rows}
            pre_existing = [i for i in seen if i in found_ids]
            if pre_existing:
                raise DuplicateObject(pre_existing[0])

    def get(self, object_id: str) -> dict[str, Any] | None:
        results = (
            self._table.search()
            .where(f"{OBJECT_ID_COLUMN} = {_quote_literal(object_id)}", prefilter=True)
            .limit(1)
            .to_list()
        )
        if not results:
            return None
        return self._row_to_object_dict(results[0])

    def batch_get(self, object_ids: list[str]) -> list[dict[str, Any] | None]:
        """Fetch multiple objects in one scan."""
        if not object_ids:
            return []
        unique = list(dict.fromkeys(object_ids))
        in_list = ", ".join(_quote_literal(oid) for oid in unique)
        rows = (
            self._table.search()
            .where(f"{OBJECT_ID_COLUMN} IN ({in_list})", prefilter=True)
            .limit(len(unique))
            .to_list()
        )
        by_id: dict[str, dict[str, Any]] = {
            row[OBJECT_ID_COLUMN]: self._row_to_object_dict(row) for row in rows
        }
        return [by_id.get(oid) for oid in object_ids]

    def _row_to_object_dict(self, row: dict[str, Any]) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        vectors: dict[str, list[float] | None] = {}
        for key, value in row.items():
            if key == OBJECT_ID_COLUMN:
                continue
            if key.startswith(VECTOR_COLUMN_PREFIX):
                name = key[len(VECTOR_COLUMN_PREFIX) :]
                vectors[name] = value
            else:
                arrow_type = self._column_type(key) if key in self._table_columns() else None
                properties[key] = self._decode_property(value, arrow_type)
        for rec in self._registry.list_vectors():
            vectors.setdefault(rec.name, None)
        return {
            "object_id": row[OBJECT_ID_COLUMN],
            "properties": properties,
            "vectors": vectors,
        }

    def _decode_property(self, value: Any, arrow_type: pa.DataType | None) -> Any:
        return value

    def delete(self, object_id: str) -> None:
        _with_retry(self._table.delete, f"{OBJECT_ID_COLUMN} = {_quote_literal(object_id)}")

    # ------------------------------------------------------------------
    # Update / batch_update
    # ------------------------------------------------------------------

    def update(
        self,
        object_id: str,
        properties: dict[str, Any] | None = None,
        vectors: dict[str, list[float] | None] | None = None,
        *,
        allow_insert: bool = False,
    ) -> None:
        self._apply_update(object_id, properties, vectors, allow_insert=allow_insert)

    def batch_update(
        self,
        updates: list[dict[str, Any]],
    ) -> None:
        """Apply a list of {object_id, properties?, vectors?} updates.

        All ids must already exist in the table; raises ObjectNotFound for any
        missing id (checked in one batch query before any writes).

        A batch spanning rows with differing column signatures executes as N
        independent merge_insert calls — each is atomic and conflict-retried;
        the batch as a whole is not. For full atomicity, issue homogeneous batches
        (same set of columns for every row).
        """
        merge_rows: list[dict[str, Any]] = []
        null_scalar_ops: list[tuple[str, str]] = []
        null_vector_ops: list[tuple[str, str]] = []

        seen: set[str] = set()
        for upd in updates:
            object_id = upd["object_id"]
            if object_id in seen:
                raise DuplicateObject(object_id)
            seen.add(object_id)
            properties = upd.get("properties") or {}
            vectors = upd.get("vectors") or {}

            write_props = {k: v for k, v in properties.items() if v is not None}
            null_props = [k for k, v in properties.items() if v is None]
            write_vecs = {k: v for k, v in vectors.items() if v is not None}
            null_vecs = [k for k, v in vectors.items() if v is None]

            if write_props or write_vecs:
                merge_rows.append(self._prepare_row(object_id, write_props, write_vecs))
            for p in null_props:
                null_scalar_ops.append((object_id, p))
            for v in null_vecs:
                null_vector_ops.append((object_id, v))

        # Batch existence check — one query for all ids.
        if seen:
            missing = self._find_missing_ids(list(seen))
            if missing:
                raise ObjectNotFound(missing[0])

        # Group rows by column signature so that when_matched_update_all() only
        # writes the columns each row actually touches.
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for row in merge_rows:
            key = tuple(sorted(row.keys()))
            groups.setdefault(key, []).append(row)
        for rows in groups.values():
            table = self._batch_from_rows(rows)
            result = (
                self._table.merge_insert(OBJECT_ID_COLUMN).when_matched_update_all().execute(table)
            )
            if result.num_updated_rows < len(rows):
                group_ids = [row[OBJECT_ID_COLUMN] for row in rows]
                missing = self._find_missing_ids(group_ids)
                if missing:
                    raise ObjectNotFound(missing[0])

        for object_id, prop in null_scalar_ops:
            result = self._clear_scalar(object_id, prop)
            if result.rows_updated == 0:
                raise ObjectNotFound(object_id)
        for object_id, vec in null_vector_ops:
            result = self._clear_vector(object_id, vec)
            if result.num_updated_rows == 0:
                raise ObjectNotFound(object_id)

    def _apply_update(
        self,
        object_id: str,
        properties: dict[str, Any] | None,
        vectors: dict[str, list[float] | None] | None,
        allow_insert: bool,
    ) -> None:
        write_props: dict[str, Any] = {}
        null_props: list[str] = []
        write_vecs: dict[str, list[float]] = {}
        null_vecs: list[str] = []
        if properties:
            for k, v in properties.items():
                if v is None:
                    null_props.append(k)
                else:
                    write_props[k] = v
        if vectors:
            for k, v in vectors.items():
                if v is None:
                    null_vecs.append(k)
                else:
                    write_vecs[k] = v

        existence_checked = False

        if write_props or write_vecs:
            row = self._prepare_row(object_id, write_props, write_vecs)
            table = self._batch_from_rows([row])
            builder = self._table.merge_insert(OBJECT_ID_COLUMN).when_matched_update_all()
            if allow_insert:
                builder = builder.when_not_matched_insert_all()
            result = builder.execute(table)
            if not allow_insert and result.num_updated_rows == 0:
                raise ObjectNotFound(object_id)
            existence_checked = True

        for prop in null_props:
            result = self._clear_scalar(object_id, prop)
            if not existence_checked:
                if not allow_insert and result.rows_updated == 0:
                    raise ObjectNotFound(object_id)
                existence_checked = True

        for vec in null_vecs:
            result = self._clear_vector(object_id, vec)
            if not existence_checked:
                if not allow_insert and result.num_updated_rows == 0:
                    raise ObjectNotFound(object_id)
                existence_checked = True

    def _clear_scalar(self, object_id: str, property_name: str):
        if property_name not in self._table_columns():
            raise SchemaError(f"Cannot clear property {property_name!r}: column does not exist.")
        arrow_type = self._column_type(property_name)
        sql_type = arrow_type_to_sql_type(arrow_type)
        return self._table.update(
            where=f"{OBJECT_ID_COLUMN} = {_quote_literal(object_id)}",
            values_sql={property_name: f"CAST(NULL AS {sql_type})"},
        )

    def _clear_vector(self, object_id: str, vector_name: str):
        rec = self._registry.get_vector(vector_name)
        if rec is None:
            raise VectorFieldNotRegistered(vector_name)
        ids = pa.array([object_id], type=pa.string())
        vec = pa.array([None], type=pa.list_(pa.float32(), rec.dim))
        batch_schema = pa.schema(
            [
                pa.field(OBJECT_ID_COLUMN, pa.string(), nullable=False),
                pa.field(rec.column, pa.list_(pa.float32(), rec.dim)),
            ]
        )
        batch = pa.RecordBatch.from_arrays([ids, vec], schema=batch_schema)
        return self._table.merge_insert(OBJECT_ID_COLUMN).when_matched_update_all().execute(batch)

    # ------------------------------------------------------------------
    # Schema evolution
    # ------------------------------------------------------------------

    def drop_fields(self, names: Iterable[str]) -> None:
        columns_to_drop: list[str] = []
        for name in names:
            rec = self._registry.get_vector(name)
            if rec is not None:
                if rec.index:
                    self._drop_index_by_column(rec.column)
                columns_to_drop.append(rec.column)
                self._registry.remove_vector(name)
                continue
            if name in self._table_columns():
                if name == OBJECT_ID_COLUMN:
                    raise SchemaError("Cannot drop the object_id column.")
                columns_to_drop.append(name)
                self._registry.remove_property(name)
        if columns_to_drop:
            _with_retry(self._table.drop_columns, columns_to_drop)

    def rename_field(self, old: str, new: str) -> None:
        rec = self._registry.get_vector(old)
        if rec is not None:
            if new.startswith(VECTOR_COLUMN_PREFIX):
                raise SchemaError(
                    f"New vector name {new!r} must not start with {VECTOR_COLUMN_PREFIX!r}."
                )
            if self._registry.has_vector(new):
                raise SchemaError(f"Vector field {new!r} already exists.")
            old_column = rec.column
            new_column = self._registry.vector_column(new)
            saved_index = rec.index
            if saved_index:
                self._drop_index_by_column(old_column)
            _with_retry(self._table.alter_columns, {"path": old_column, "rename": new_column})
            new_rec = self._registry.rename_vector(old, new)
            if saved_index:
                self._recreate_index(new_rec, saved_index)
            return
        if old in self._table_columns():
            if old == OBJECT_ID_COLUMN:
                raise SchemaError("Cannot rename the object_id column.")
            if new.startswith(VECTOR_COLUMN_PREFIX):
                raise SchemaError(
                    f"Property name {new!r} must not start with {VECTOR_COLUMN_PREFIX!r}."
                )
            if new in self._table_columns():
                raise SchemaError(f"Column {new!r} already exists.")
            _with_retry(self._table.alter_columns, {"path": old, "rename": new})
            self._registry.rename_property(old, new)
            return
        raise SchemaError(f"No field named {old!r} to rename.")

    def schema(self) -> dict[str, dict[str, Any]]:
        properties: dict[str, str] = {}
        vectors: dict[str, dict[str, Any]] = {}
        for f in self._table.schema:
            if f.name == OBJECT_ID_COLUMN:
                continue
            if f.name.startswith(VECTOR_COLUMN_PREFIX):
                continue
            properties[f.name] = str(f.type)
        for rec in self._registry.list_vectors():
            vectors[rec.name] = {"dim": rec.dim}
            if rec.description:
                vectors[rec.name]["description"] = rec.description
        return {"properties": properties, "vectors": vectors}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        vector_field: str,
        limit: int = 10,
        metric: str | None = None,
        where: str | None = None,
        select: list[str] | None = None,
        nprobes: int | None = None,
        refine_factor: int | None = None,
    ) -> list[SearchResult]:
        rec, effective_metric = self._prepare_vector_query(query_vector, vector_field, metric)

        builder = self._table.search(query_vector, vector_column_name=rec.column)
        builder = builder.distance_type(effective_metric)
        if where:
            builder = builder.where(where, prefilter=True)
        if select:
            user_select = list(dict.fromkeys([OBJECT_ID_COLUMN, *select]))
            builder = builder.select(user_select)
        builder = builder.limit(limit)
        if nprobes is not None:
            builder = builder.nprobes(nprobes)
        if refine_factor is not None:
            builder = builder.refine_factor(refine_factor)
        rows = builder.to_list()

        return [self._row_to_result(row, effective_metric, select) for row in rows]

    def search_within(
        self,
        query_vector: list[float],
        vector_field: str,
        max_distance: float,
        min_distance: float | None = None,
        limit: int | None = None,
        metric: str | None = None,
        where: str | None = None,
        select: list[str] | None = None,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        exact: bool = False,
    ) -> list[SearchResult]:
        self._validate_distance_bounds(min_distance, max_distance)
        rec, effective_metric = self._prepare_vector_query(query_vector, vector_field, metric)

        lower = None if min_distance is None else float(min_distance)
        upper = float(max_distance)

        builder = self._table.search(query_vector, vector_column_name=rec.column)
        builder = builder.distance_type(effective_metric)
        builder = builder.distance_range(lower, upper)
        if exact:
            builder = builder.bypass_vector_index()
        if where:
            builder = builder.where(where, prefilter=True)
        if select:
            user_select = list(dict.fromkeys([OBJECT_ID_COLUMN, *select]))
            builder = builder.select(user_select)
        builder = builder.limit(_UNBOUNDED_LIMIT if limit is None else int(limit))
        if nprobes is not None:
            builder = builder.nprobes(nprobes)
        if refine_factor is not None:
            builder = builder.refine_factor(refine_factor)
        rows = builder.to_list()

        return [self._row_to_result(row, effective_metric, select) for row in rows]

    def _prepare_vector_query(
        self,
        query_vector: list[float],
        vector_field: str,
        metric: str | None,
    ) -> tuple[VectorFieldRecord, str]:
        """Validate inputs and resolve the effective metric for a vector query."""
        if query_vector is None or len(query_vector) == 0:
            raise ValueError("Vector query requires a non-empty query_vector.")
        rec = self._registry.get_vector(vector_field)
        if rec is None:
            raise VectorFieldNotRegistered(vector_field)
        if len(query_vector) != rec.dim:
            raise DimensionMismatch(vector_field, rec.dim, len(query_vector))

        effective_metric = normalize_metric(metric) if metric else None
        if rec.index and rec.index.get("metric"):
            index_metric = rec.index["metric"]
            if effective_metric is not None and effective_metric != index_metric:
                raise MetricMismatch(vector_field, effective_metric, index_metric)
            effective_metric = index_metric
        if effective_metric is None:
            effective_metric = DEFAULT_METRIC
        return rec, effective_metric

    @staticmethod
    def _validate_distance_bounds(min_distance: float | None, max_distance: float) -> None:
        if max_distance is None:
            raise ValueError("search_within() requires max_distance.")
        if not math.isfinite(max_distance):
            raise ValueError(f"max_distance must be a finite number, got {max_distance!r}.")
        if min_distance is not None:
            if not math.isfinite(min_distance):
                raise ValueError(f"min_distance must be a finite number, got {min_distance!r}.")
            if min_distance >= max_distance:
                raise ValueError(
                    f"min_distance ({min_distance}) must be strictly less than "
                    f"max_distance ({max_distance})."
                )

    @staticmethod
    def _row_to_result(
        row: dict[str, Any],
        effective_metric: str,
        select: list[str] | None,
    ) -> SearchResult:
        distance = row.pop("_distance")
        object_id = row.pop(OBJECT_ID_COLUMN)
        row.pop("_rowid", None)
        row.pop("_score", None)
        properties = {k: v for k, v in row.items() if not k.startswith(VECTOR_COLUMN_PREFIX)}
        if select is not None:
            properties = {k: v for k, v in properties.items() if k in set(select)}
        return SearchResult(
            object_id=object_id,
            score=distance_to_score(distance, effective_metric),
            properties=properties,
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_index(
        self,
        vector_field: str,
        index_type: str = "IVF_PQ",
        metric: str = DEFAULT_METRIC,
        replace: bool = True,
        **params: Any,
    ) -> None:
        rec = self._registry.get_vector(vector_field)
        if rec is None:
            raise VectorFieldNotRegistered(vector_field)
        metric_n = normalize_metric(metric)
        _with_retry(
            self._table.create_index,
            metric=metric_n,
            vector_column_name=rec.column,
            index_type=index_type,
            replace=replace,
            **params,
        )
        self._registry.set_index(
            vector_field,
            {"type": index_type, "metric": metric_n, **params},
        )

    def rebuild_index(self, vector_field: str) -> None:
        rec = self._registry.get_vector(vector_field)
        if rec is None:
            raise VectorFieldNotRegistered(vector_field)
        if not rec.index:
            raise SchemaError(
                f"Vector field {vector_field!r} has no index to rebuild. Call create_index first."
            )
        saved = dict(rec.index)
        self._drop_index_by_column(rec.column)
        self._recreate_index(rec, saved)

    def drop_index(self, vector_field: str) -> None:
        rec = self._registry.get_vector(vector_field)
        if rec is None:
            raise VectorFieldNotRegistered(vector_field)
        if not rec.index:
            return
        self._drop_index_by_column(rec.column)
        self._registry.set_index(vector_field, None)

    def index_info(self, vector_field: str) -> IndexInfo | None:
        rec = self._registry.get_vector(vector_field)
        if rec is None:
            raise VectorFieldNotRegistered(vector_field)
        if not rec.index:
            return None
        params = {k: v for k, v in rec.index.items() if k not in {"type", "metric"}}
        return IndexInfo(
            vector_field=vector_field,
            index_type=rec.index["type"],
            metric=rec.index["metric"],
            params=params,
        )

    def _drop_index_by_column(self, column: str) -> None:
        for idx in self._table.list_indices():
            if column in idx.columns:
                try:
                    self._table.drop_index(idx.name)
                except Exception as exc:  # pragma: no cover
                    log.warning("Failed to drop index %s: %s", idx.name, exc)

    def _recreate_index(self, rec: VectorFieldRecord, saved: dict[str, Any]) -> None:
        params = {k: v for k, v in saved.items() if k not in {"type", "metric"}}
        _with_retry(
            self._table.create_index,
            metric=saved["metric"],
            vector_column_name=rec.column,
            index_type=saved["type"],
            replace=True,
            **params,
        )
        self._registry.set_index(rec.name, saved)

    # ------------------------------------------------------------------
    # Bulk read
    # ------------------------------------------------------------------

    def export_vectors(
        self, vector_field: str, where: str | None = None
    ) -> tuple[list[str], np.ndarray]:
        rec = self._registry.get_vector(vector_field)
        if rec is None:
            raise VectorFieldNotRegistered(vector_field)
        total = self._table.count_rows(where) if where else self._table.count_rows()
        if total == 0:
            return [], np.zeros((0, rec.dim), dtype=np.float32)
        builder = self._table.search().select([OBJECT_ID_COLUMN, rec.column]).limit(total)
        if where:
            builder = builder.where(where, prefilter=True)
        arrow_table = builder.to_arrow()
        ids_col = arrow_table[OBJECT_ID_COLUMN].to_pylist()
        vec_col = arrow_table[rec.column].to_pylist()
        kept_ids: list[str] = []
        kept_vecs: list[list[float]] = []
        for oid, vec in zip(ids_col, vec_col, strict=True):
            if vec is None:
                continue
            kept_ids.append(oid)
            kept_vecs.append(vec)
        if not kept_vecs:
            return kept_ids, np.zeros((0, rec.dim), dtype=np.float32)
        return kept_ids, np.asarray(kept_vecs, dtype=np.float32)

    def optimize(self) -> None:
        self._table.optimize()

    def list_objects(
        self,
        where: str | None = None,
        select: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        builder = self._table.search()
        if where:
            builder = builder.where(where, prefilter=True)
        if select is not None:
            cols = list(dict.fromkeys([OBJECT_ID_COLUMN, *select]))
            builder = builder.select(cols)
        if limit is not None:
            builder = builder.limit(limit)
        if offset is not None:
            builder = builder.offset(offset)
        rows = builder.to_list()
        out: list[dict[str, Any]] = []
        for row in rows:
            row.pop("_distance", None)
            row.pop("_rowid", None)
            row.pop("_score", None)
            obj = self._row_to_object_dict(row)
            if select is not None:
                obj["properties"] = {k: v for k, v in obj["properties"].items() if k in set(select)}
            out.append(obj)
        return out
