"""object_store — Multimodal object store on top of LanceDB."""

from __future__ import annotations

from .exceptions import (
    DimensionMismatch,
    DuplicateObject,
    MetricMismatch,
    ObjectNotFound,
    ObjectStoreError,
    SchemaError,
    VectorFieldNotRegistered,
)
from .store import ObjectStore
from .types import (
    IndexInfo,
    ObjectData,
    ObjectUpdate,
    SearchResult,
    VectorFieldInfo,
)

__all__ = [
    "DimensionMismatch",
    "DuplicateObject",
    "IndexInfo",
    "MetricMismatch",
    "ObjectData",
    "ObjectNotFound",
    "ObjectStore",
    "ObjectStoreError",
    "ObjectUpdate",
    "SchemaError",
    "SearchResult",
    "VectorFieldInfo",
    "VectorFieldNotRegistered",
]
