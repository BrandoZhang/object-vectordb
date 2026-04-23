"""object_vectordb — Multimodal object-centric vector database on top of LanceDB."""

from __future__ import annotations

from .collection import Collection
from .db import ObjectVectorDB
from .exceptions import (
    DimensionMismatch,
    DuplicateObject,
    MetricMismatch,
    ObjectNotFound,
    ObjectVectorDBError,
    SchemaError,
    VectorFieldNotRegistered,
)
from .fusion import rrf_merge
from .types import (
    IndexInfo,
    ObjectAdd,
    ObjectData,
    ObjectUpdate,
    OnMissing,
    SearchResult,
    VectorFieldInfo,
)

__all__ = [
    "Collection",
    "DimensionMismatch",
    "DuplicateObject",
    "IndexInfo",
    "MetricMismatch",
    "ObjectAdd",
    "ObjectData",
    "ObjectNotFound",
    "ObjectUpdate",
    "ObjectVectorDB",
    "ObjectVectorDBError",
    "OnMissing",
    "SchemaError",
    "SearchResult",
    "VectorFieldInfo",
    "VectorFieldNotRegistered",
    "rrf_merge",
]
