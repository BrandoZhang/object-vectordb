"""Public `ObjectVectorDB` class.

This module must not import `lancedb` or `pyarrow`. All backend-specific operations
go through `LanceDBBackend`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .backend import LanceDBBackend
from .registry import SchemaRegistry
from .types import IndexInfo, ObjectData, ObjectUpdate, SearchResult, VectorFieldInfo


class ObjectVectorDB:
    """Object-centric multimodal store built on LanceDB.

    Example:
        store = ObjectVectorDB(uri="data/media", table_name="videos")
        store.register_vector_field("clip", dim=512)
        store.add("v1", properties={"title": "t"}, vectors={"clip": [...]})
        results = store.search([...], vector_field="clip", limit=10)
    """

    def __init__(
        self,
        uri: str,
        table_name: str = "objects",
        auto_register: bool = False,
    ):
        self._uri = uri
        self._table_name = table_name
        self._registry = SchemaRegistry(uri)
        self._backend = LanceDBBackend(
            uri=uri,
            table_name=table_name,
            registry=self._registry,
            auto_register=auto_register,
        )

    # ------------------------------------------------------------------
    # Vector field registration
    # ------------------------------------------------------------------

    def register_vector_field(
        self, name: str, dim: int, description: str | None = None
    ) -> VectorFieldInfo:
        return self._backend.register_vector_field(name, dim, description)

    def vector_fields(self) -> list[VectorFieldInfo]:
        return self._backend.list_vector_fields()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        object_id: str,
        properties: dict[str, Any] | None = None,
        vectors: dict[str, list[float] | None] | None = None,
    ) -> None:
        self._backend.add(object_id, properties, vectors)

    def add_many(self, items: list[dict[str, Any]]) -> None:
        self._backend.add_many(items)

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

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        object_id: str,
        properties: dict[str, Any] | None = None,
        vectors: dict[str, list[float] | None] | None = None,
    ) -> None:
        self._backend.update(object_id, properties, vectors)

    def batch_update(self, updates: Iterable[ObjectUpdate]) -> None:
        payload = [
            {
                "object_id": upd.object_id,
                "properties": upd.properties,
                "vectors": upd.vectors,
            }
            for upd in updates
        ]
        self._backend.batch_update(payload)

    # ------------------------------------------------------------------
    # Schema evolution
    # ------------------------------------------------------------------

    def drop_fields(self, names: Iterable[str]) -> None:
        self._backend.drop_fields(names)

    def rename_field(self, old: str, new: str) -> None:
        self._backend.rename_field(old, new)

    def schema(self) -> dict[str, dict[str, Any]]:
        return self._backend.schema()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

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

    @staticmethod
    def rrf_merge(
        *result_lists: list[SearchResult], k: int = 60, limit: int | None = None
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion over multiple SearchResult lists.

        For each object id, the fused score is sum over input lists of 1/(k + rank),
        where rank is 1-indexed position in that list. Objects appearing in only one
        list still get a score. Properties from the first occurrence are preserved.
        """
        scores: dict[str, float] = {}
        props: dict[str, dict[str, Any]] = {}
        first_seen: dict[str, int] = {}
        counter = 0
        for results in result_lists:
            for rank, hit in enumerate(results, start=1):
                scores[hit.object_id] = scores.get(hit.object_id, 0.0) + 1.0 / (k + rank)
                if hit.object_id not in props:
                    props[hit.object_id] = hit.properties
                    first_seen[hit.object_id] = counter
                    counter += 1
        ranked = sorted(
            scores.items(),
            key=lambda kv: (-kv[1], first_seen[kv[0]]),
        )
        if limit is not None:
            ranked = ranked[:limit]
        return [
            SearchResult(object_id=oid, score=score, properties=props[oid]) for oid, score in ranked
        ]

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Bulk read
    # ------------------------------------------------------------------

    def export_vectors(
        self, vector_field: str, where: str | None = None
    ) -> tuple[list[str], np.ndarray]:
        return self._backend.export_vectors(vector_field, where=where)

    def list(
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
