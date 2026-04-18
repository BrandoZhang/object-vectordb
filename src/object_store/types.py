"""Public dataclasses returned by the ObjectStore API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObjectData:
    """An object as returned by `ObjectStore.get()`."""

    object_id: str
    properties: dict[str, Any]
    vectors: dict[str, list[float] | None]


@dataclass
class SearchResult:
    """A single hit from `ObjectStore.search()`.

    `score` is a similarity score where higher means more similar.
    - For metric="cosine": score = 1 - cosine_distance, range [-1, 1].
    - For metric="l2":     score = 1 / (1 + l2_distance), range (0, 1]; monotonic only,
      magnitudes are not physically meaningful, do not compare across different query vectors.
    - For metric="dot":    score = 1 - (1 - dot_product) = dot_product, range unbounded.
    """

    object_id: str
    score: float
    properties: dict[str, Any]


@dataclass
class ObjectUpdate:
    """Batch-update entry for `ObjectStore.batch_update()`."""

    object_id: str
    properties: dict[str, Any] | None = None
    vectors: dict[str, list[float] | None] | None = None


@dataclass
class VectorFieldInfo:
    """Information about a registered vector field."""

    name: str
    dim: int
    has_index: bool
    description: str | None = None


@dataclass
class IndexInfo:
    """Information about an index on a vector field."""

    vector_field: str
    index_type: str
    metric: str
    params: dict[str, Any] = field(default_factory=dict)
