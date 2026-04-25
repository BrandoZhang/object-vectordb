"""Public `ObjectVectorDB` class — the top-level DB handle.

`ObjectVectorDB(uri)` opens a LanceDB directory.  Collections are created/opened
via `db.collection(name)` and are the unit where schemas, vector fields, and
indexes live.

This module must not import `pyarrow`; table creation lives inside the backend.
`lancedb.connect` is the only storage-engine call here.
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

    def __init__(self, uri: str, storage_options: dict[str, str] | None = None):
        """Open (or lazily create) a LanceDB directory.

        `storage_options` is forwarded to `lancedb.connect` and is the only
        way to enable Lance's multi-writer commit coordination on object
        stores.  For S3 with concurrent writers, Lance requires either an
        external commit lock (e.g. DynamoDB) or server-side conditional PUT;
        without one of these, concurrent manifest commits silently clobber
        each other and rows / schema changes are lost.  See
        docs/concurrency.md and the LanceDB storage docs for the current
        option names.
        """
        self._uri = uri
        connect_kwargs: dict[str, object] = {}
        if storage_options is not None:
            connect_kwargs["storage_options"] = storage_options
        self._db = lancedb.connect(uri, **connect_kwargs)
        self._registry = SchemaRegistry(uri, self._db)

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
        """Return collection names at this URI.

        Uses db.table_names() filtered by the ovdb sentinel on the object_id
        field; tables created outside of object_vectordb are excluded.
        """
        return self._registry.list_collections()

    def has_collection(self, name: str) -> bool:
        return self._registry.has_collection(name)

    def drop_collection(self, name: str) -> None:
        """Drop a collection: delete the Lance table.

        Silent no-op if the collection does not exist.
        """
        if name in self._db.table_names():
            self._db.drop_table(name)
