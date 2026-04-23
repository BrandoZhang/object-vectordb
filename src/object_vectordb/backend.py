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
from .registry import VECTOR_COLUMN_PREFIX, CollectionRegistry, VectorFieldRecord
from .scoring import distance_to_score, normalize_metric
from .types import IndexInfo, OnMissing, SearchResult, VectorFieldInfo

_VALID_ON_MISSING: frozenset[str] = frozenset({"raise", "insert", "skip"})


def _validate_on_missing(value: str) -> None:
    if value not in _VALID_ON_MISSING:
        raise ValueError(f"on_missing must be one of {sorted(_VALID_ON_MISSING)!r}, got {value!r}")


log = logging.getLogger(__name__)

OBJECT_ID_COLUMN = "object_id"
DEFAULT_METRIC = "cosine"
# LanceDB's builder requires a limit; this sentinel is used when the public API
# says "unbounded" (search_within with limit=None).
_UNBOUNDED_LIMIT = 2**31 - 1


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

    # ------------------------------------------------------------------
    # Table bootstrap
    # ------------------------------------------------------------------

    def _ensure_table(self):
        if self._table_name in self._db.table_names():
            return self._db.open_table(self._table_name)
        schema = pa.schema([pa.field(OBJECT_ID_COLUMN, pa.string(), nullable=False)])
        return self._db.create_table(self._table_name, schema=schema)

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
            if not self._registry.has_property(name):
                self._registry.add_property(name)
            return
        dtype = python_value_to_arrow_type(sample)
        self._table.add_columns(pa.field(name, dtype))
        self._registry.add_property(name)

    def _ensure_vector_column(self, name: str, dim: int) -> VectorFieldRecord:
        column = self._registry.vector_column(name)
        existing = self._registry.get_vector(name)
        if existing is not None:
            if existing.dim != dim:
                raise DimensionMismatch(name, existing.dim, dim)
            return existing
        if column in self._table_columns():
            # Column exists in the lance table but not in the registry — re-register.
            return self._registry.add_vector(name, dim)
        self._table.add_columns(pa.field(column, pa.list_(pa.float32(), dim)))
        return self._registry.add_vector(name, dim)

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
                existing.description = description
                self._registry.set_index(name, existing.index)  # triggers save
            return self._vector_field_info(existing)
        column = self._registry.vector_column(name)
        if column not in self._table_columns():
            self._table.add_columns(pa.field(column, pa.list_(pa.float32(), dim)))
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
        """Build a row dict ready for pyarrow from the user's Python inputs.

        - Ensures every property column exists (auto-add) and JSON-encodes dicts.
        - Validates/auto-registers vector fields and maps names to __vec_ columns.
        """
        row: dict[str, Any] = {OBJECT_ID_COLUMN: object_id}
        if properties:
            for key, value in properties.items():
                if value is None:
                    # Can't infer type; only allowed if column already exists
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
        """Build a pyarrow Table whose schema contains exactly the columns present in `rows`.

        LanceDB's merge_insert accepts a subset of the target's columns as long as the key is
        present; unspecified columns are preserved by when_matched_update_all.
        """
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

    def add(
        self,
        object_id: str,
        properties: dict[str, Any] | None,
        vectors: dict[str, list[float] | None] | None,
    ) -> None:
        if self.exists(object_id):
            raise DuplicateObject(object_id)
        row = self._prepare_row(object_id, properties, vectors)
        table = self._batch_from_rows([row])
        # Use merge_insert with only-not-matched so we never clobber a concurrent write
        # of the same id (single-writer, but defensive).
        self._table.merge_insert(OBJECT_ID_COLUMN).when_not_matched_insert_all().execute(table)

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
            if self.exists(object_id):
                raise DuplicateObject(object_id)
            rows.append(self._prepare_row(object_id, item.get("properties"), item.get("vectors")))
        table = self._batch_from_rows(rows)
        self._table.merge_insert(OBJECT_ID_COLUMN).when_not_matched_insert_all().execute(table)

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
        # Ensure all registered vector fields appear in the output (even if None).
        for rec in self._registry.list_vectors():
            vectors.setdefault(rec.name, None)
        return {
            "object_id": row[OBJECT_ID_COLUMN],
            "properties": properties,
            "vectors": vectors,
        }

    def _decode_property(self, value: Any, arrow_type: pa.DataType | None) -> Any:
        # We encode dicts as JSON strings on write. We don't know whether a given string
        # column was originally a dict, so we leave decoding to the caller.
        return value

    def delete(self, object_id: str) -> None:
        # LanceDB silently ignores non-matching deletes. Silent no-op on missing is the spec.
        self._table.delete(f"{OBJECT_ID_COLUMN} = {_quote_literal(object_id)}")

    # ------------------------------------------------------------------
    # Update / batch_update
    # ------------------------------------------------------------------

    def update(
        self,
        object_id: str,
        properties: dict[str, Any] | None = None,
        vectors: dict[str, list[float] | None] | None = None,
        on_missing: OnMissing = "raise",
    ) -> None:
        _validate_on_missing(on_missing)
        if on_missing != "insert" and not self.exists(object_id):
            if on_missing == "raise":
                raise ObjectNotFound(object_id)
            # on_missing == "skip"
            return
        self._apply_update(object_id, properties, vectors, allow_insert=on_missing == "insert")

    def batch_update(
        self,
        updates: list[dict[str, Any]],
        on_missing: OnMissing = "raise",
    ) -> None:
        """Apply a list of {object_id, properties?, vectors?} updates.

        We build one merge_insert batch per call for rows with non-None property/vector writes,
        then handle None-clears individually (they need per-column SQL casts or per-row null batches).

        `on_missing` controls behavior for ids not present in the table: "raise" (default)
        raises `ObjectNotFound` on the first missing id; "skip" silently drops missing rows;
        "insert" upserts (a partial row with only the touched columns is inserted).
        """
        _validate_on_missing(on_missing)
        merge_rows: list[dict[str, Any]] = []
        null_scalar_ops: list[tuple[str, str]] = []  # (object_id, property_name)
        null_vector_ops: list[tuple[str, str]] = []  # (object_id, vector_name)

        seen: set[str] = set()
        for upd in updates:
            object_id = upd["object_id"]
            # LanceDB's merge_insert against multiple source rows sharing a key is
            # implementation-defined, and splitting rows across signature groups
            # makes apply-order unreliable. Match batch_add's behavior and reject.
            if object_id in seen:
                raise DuplicateObject(object_id)
            seen.add(object_id)
            if on_missing != "insert" and not self.exists(object_id):
                if on_missing == "raise":
                    raise ObjectNotFound(object_id)
                # on_missing == "skip"
                continue
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

        # Group rows by column signature so that merge_insert.when_matched_update_all()
        # only writes the columns each row actually touches (otherwise missing columns
        # would be nulled out for rows in the same batch that didn't specify them).
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for row in merge_rows:
            key = tuple(sorted(row.keys()))
            groups.setdefault(key, []).append(row)
        for rows in groups.values():
            table = self._batch_from_rows(rows)
            builder = self._table.merge_insert(OBJECT_ID_COLUMN).when_matched_update_all()
            if on_missing == "insert":
                builder = builder.when_not_matched_insert_all()
            builder.execute(table)

        for object_id, prop in null_scalar_ops:
            self._clear_scalar(object_id, prop)
        for object_id, vec in null_vector_ops:
            self._clear_vector(object_id, vec)

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

        if write_props or write_vecs:
            row = self._prepare_row(object_id, write_props, write_vecs)
            table = self._batch_from_rows([row])
            builder = self._table.merge_insert(OBJECT_ID_COLUMN).when_matched_update_all()
            if allow_insert:
                builder = builder.when_not_matched_insert_all()
            builder.execute(table)
        for prop in null_props:
            self._clear_scalar(object_id, prop)
        for vec in null_vecs:
            self._clear_vector(object_id, vec)

    def _clear_scalar(self, object_id: str, property_name: str) -> None:
        # Ensure the column exists (caller may be clearing a pre-existing but unused column).
        if property_name not in self._table_columns():
            raise SchemaError(f"Cannot clear property {property_name!r}: column does not exist.")
        arrow_type = self._column_type(property_name)
        sql_type = arrow_type_to_sql_type(arrow_type)
        self._table.update(
            where=f"{OBJECT_ID_COLUMN} = {_quote_literal(object_id)}",
            values_sql={property_name: f"CAST(NULL AS {sql_type})"},
        )

    def _clear_vector(self, object_id: str, vector_name: str) -> None:
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
        self._table.merge_insert(OBJECT_ID_COLUMN).when_matched_update_all().execute(batch)

    # ------------------------------------------------------------------
    # Schema evolution
    # ------------------------------------------------------------------

    def drop_fields(self, names: Iterable[str]) -> None:
        columns_to_drop: list[str] = []
        for name in names:
            rec = self._registry.get_vector(name)
            if rec is not None:
                # Drop any index first.
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
            self._table.drop_columns(columns_to_drop)

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
            self._table.alter_columns({"path": old_column, "rename": new_column})
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
            self._table.alter_columns({"path": old, "rename": new})
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

        # Pass lower_bound=None (not 0.0) when the caller did not specify one:
        # for metric="dot" the LanceDB distance is `1 - dot(q,v)` and can be
        # negative, so a 0.0 floor would silently drop valid matches.
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
        # LanceDB requires a limit; None means "unbounded" at our API boundary.
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
        # Negative `max_distance` is allowed for metric="dot" (LanceDB distance
        # is `1 - dot` and can be negative). For cosine/l2 a non-positive bound
        # will simply return an empty result — caller's responsibility.
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
        # Filter to requested properties; strip internal vector columns and _rowid, _score.
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
        self._table.create_index(
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
        self._table.create_index(
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
            # list() returns the flattened per-spec — for the public wrapper it's easier
            # to expose {object_id, properties} without vectors by default.
            if select is not None:
                obj["properties"] = {k: v for k, v in obj["properties"].items() if k in set(select)}
            out.append(obj)
        return out
