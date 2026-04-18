"""object_vectordb — Multimodal object-centric vector database on top of LanceDB."""

from __future__ import annotations

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
    "ObjectUpdate",
    "ObjectVectorDB",
    "ObjectVectorDBError",
    "SchemaError",
    "SearchResult",
    "VectorFieldInfo",
    "VectorFieldNotRegistered",
]
