"""Public `ObjectVectorDB` class — the top-level DB handle.

`ObjectVectorDB(uri)` opens a LanceDB directory and a schema-registry sidecar.
Collections are created/opened via `db.collection(name)` and are the unit
where schemas, vector fields, and indexes live.

This module must not import `pyarrow`; table creation lives inside the
backend. `lancedb.connect` is the only storage-engine call here.
"""

from __future__ import annotations

import lancedb

from .collection import Collection
from .registry import SchemaRegistry


class ObjectVectorDB:
    """A handle to a LanceDB directory that hosts one or more `Collection`s.

    Example:
        db = ObjectVectorDB(uri="data/media")
        videos = db.collection("videos")
        images = db.collection("images", auto_register=True)
        db.list_collections()   # ["images", "videos"]
        db.drop_collection("images")
    """

    def __init__(self, uri: str):
        self._uri = uri
        self._registry = SchemaRegistry(uri)
        self._db = lancedb.connect(uri)

    def collection(self, name: str, auto_register: bool = False) -> Collection:
        """Open (or create) a collection by name.

        The underlying Lance table is created on first access; vector and
        property columns are added later via `register_vector_field` and the
        first write, respectively.

        `auto_register` is a per-call flag — not persisted. It tells the
        returned Collection to implicitly register vector fields on first
        write instead of raising `VectorFieldNotRegistered`.
        """
        if not name:
            raise ValueError("Collection name must be a non-empty string.")
        col_registry = self._registry.collection(name)
        return Collection(
            db_connection=self._db,
            name=name,
            registry=col_registry,
            auto_register=auto_register,
        )

    def list_collections(self) -> list[str]:
        """Return collection names registered at this URI.

        Intersects the registry with the actual Lance tables; a registry entry
        without a corresponding table (e.g. after an external drop) is filtered
        out.
        """
        tables = set(self._db.table_names())
        return sorted(n for n in self._registry.list_collections() if n in tables)

    def has_collection(self, name: str) -> bool:
        return name in self._db.table_names() and self._registry.has_collection(name)

    def drop_collection(self, name: str) -> None:
        """Drop a collection: delete the Lance table and its registry entry.

        Silent no-op if the collection does not exist.
        """
        if name in self._db.table_names():
            self._db.drop_table(name)
        self._registry.drop_collection(name)
