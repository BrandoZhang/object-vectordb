"""Public `Collection` class.

A Collection is a named group of objects within an `ObjectVectorDB`. Collections
own their own vector-field registry, their own property schema, and their own
Lance table. Operations like `search`, `register_vector_field`, and `add` only
affect the collection they're called on; other collections on the same DB are
isolated.

This module must not import `lancedb` or `pyarrow`. All backend-specific operations
go through `LanceDBBackend`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from .backend import LanceDBBackend
from .registry import CollectionRegistry
from .types import IndexInfo, ObjectData, ObjectUpdate, OnMissing, SearchResult, VectorFieldInfo

if TYPE_CHECKING:
    pass


class Collection:
    """A named collection of objects with isolated schema and vector fields.

    Obtained via `ObjectVectorDB(uri).collection(name)`. All per-object operations
    (add, get, search, register_vector_field, etc.) live here.

    Example:
        db = ObjectVectorDB(uri="data/media")
        videos = db.collection("videos")
        videos.register_vector_field("clip", dim=512)
        videos.add("v1", properties={"title": "t"}, vectors={"clip": [...]})
        hits = videos.search([...], vector_field="clip", limit=10)
    """

    def __init__(
        self,
        db_connection,
        name: str,
        registry: CollectionRegistry,
        auto_register: bool = False,
    ):
        self._name = name
        self._backend = LanceDBBackend(
            db=db_connection,
            table_name=name,
            registry=registry,
            auto_register=auto_register,
        )

    @property
    def name(self) -> str:
        return self._name

    def register_vector_field(
        self, name: str, dim: int, description: str | None = None
    ) -> VectorFieldInfo:
        return self._backend.register_vector_field(name, dim, description)

    def list_vector_fields(self) -> list[VectorFieldInfo]:
        return self._backend.list_vector_fields()

    def add(
        self,
        object_id: str,
        properties: dict[str, Any] | None = None,
        vectors: dict[str, list[float] | None] | None = None,
    ) -> None:
        self._backend.add(object_id, properties, vectors)

    def batch_add(self, items: list[dict[str, Any]]) -> None:
        self._backend.batch_add(items)

    def get(self, object_id: str) -> ObjectData | None:
        row = self._backend.get(object_id)
        if row is None:
            return None
        return ObjectData(
            object_id=row["object_id"],
            properties=row["properties"],
            vectors=row["vectors"],
        )

    def exists(self, object_id: str) -> bool:
        return self._backend.exists(object_id)

    def delete(self, object_id: str) -> None:
        self._backend.delete(object_id)

    def update(
        self,
        object_id: str,
        properties: dict[str, Any] | None = None,
        vectors: dict[str, list[float] | None] | None = None,
        on_missing: OnMissing = "raise",
    ) -> None:
        self._backend.update(object_id, properties, vectors, on_missing=on_missing)

    def batch_update(
        self,
        updates: Iterable[ObjectUpdate],
        on_missing: OnMissing = "raise",
    ) -> None:
        payload = [
            {
                "object_id": upd.object_id,
                "properties": upd.properties,
                "vectors": upd.vectors,
            }
            for upd in updates
        ]
        self._backend.batch_update(payload, on_missing=on_missing)

    def drop_fields(self, names: Iterable[str]) -> None:
        self._backend.drop_fields(names)

    def rename_field(self, old: str, new: str) -> None:
        self._backend.rename_field(old, new)

    def schema(self) -> dict[str, dict[str, Any]]:
        return self._backend.schema()

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        vector_field: str,
        limit: int = 10,
        metric: str | None = None,
        where: str | None = None,
        select: list[str] | None = None,
        nprobes: int | None = None,
        refine_factor: int | None = None,
    ) -> list[SearchResult]:
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.astype(np.float32).tolist()
        return self._backend.search(
            query_vector=query_vector,
            vector_field=vector_field,
            limit=limit,
            metric=metric,
            where=where,
            select=select,
            nprobes=nprobes,
            refine_factor=refine_factor,
        )

    def search_within(
        self,
        query_vector: list[float] | np.ndarray,
        vector_field: str,
        max_distance: float,
        *,
        min_distance: float | None = None,
        limit: int | None = None,
        metric: str | None = None,
        where: str | None = None,
        select: list[str] | None = None,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        exact: bool = False,
    ) -> list[SearchResult]:
        """Radius (distance-bounded) vector search.

        Returns every object whose LanceDB `_distance` from `query_vector` lies
        in the half-open interval `[min_distance or 0.0, max_distance)`. Results
        are sorted by ascending distance (= descending `SearchResult.score`),
        matching the `search()` ordering.

        `max_distance` is in LanceDB's native distance space for the resolved
        metric:

        - `cosine`: `1 - cosine_similarity` (0 identical, 1 orthogonal, 2 opposite)
        - `l2`:     squared Euclidean distance
        - `dot`:    `1 - dot_product`

        To convert a desired `SearchResult.score` threshold into `max_distance`:
        for `cosine` / `dot`, pass `max_distance = 1 - min_score`; for `l2`,
        pass `max_distance = (1 / min_score) - 1` (since `score = 1 / (1 + d)`).

        `limit=None` (the default) means unbounded. Other args behave the same
        as `search()`. `exact=True` disables the ANN index for this query so
        the radius scan is guaranteed to find every match — IVF indexes make
        radius queries approximate (matches in unprobed partitions are missed).
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.astype(np.float32).tolist()
        return self._backend.search_within(
            query_vector=query_vector,
            vector_field=vector_field,
            max_distance=max_distance,
            min_distance=min_distance,
            limit=limit,
            metric=metric,
            where=where,
            select=select,
            nprobes=nprobes,
            refine_factor=refine_factor,
            exact=exact,
        )

    def create_index(
        self,
        vector_field: str,
        index_type: str = "IVF_PQ",
        metric: str = "cosine",
        replace: bool = True,
        **params: Any,
    ) -> None:
        self._backend.create_index(
            vector_field=vector_field,
            index_type=index_type,
            metric=metric,
            replace=replace,
            **params,
        )

    def rebuild_index(self, vector_field: str) -> None:
        self._backend.rebuild_index(vector_field)

    def drop_index(self, vector_field: str) -> None:
        self._backend.drop_index(vector_field)

    def index_info(self, vector_field: str) -> IndexInfo | None:
        return self._backend.index_info(vector_field)

    def export_vectors(
        self, vector_field: str, where: str | None = None
    ) -> tuple[list[str], np.ndarray]:
        return self._backend.export_vectors(vector_field, where=where)

    def list_objects(
        self,
        where: str | None = None,
        select: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[ObjectData]:
        rows = self._backend.list_objects(where=where, select=select, limit=limit, offset=offset)
        return [
            ObjectData(
                object_id=r["object_id"],
                properties=r["properties"],
                vectors=r["vectors"],
            )
            for r in rows
        ]

    def optimize(self) -> None:
        """Run LanceDB's compact/optimize (file compaction + incremental index update).

        Call periodically after large batches of writes.
        """
        self._backend.optimize()
