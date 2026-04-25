"""Custom exceptions for the object_vectordb library."""

from __future__ import annotations


class ObjectVectorDBError(Exception):
    """Base class for all object_vectordb errors."""


class ObjectNotFound(ObjectVectorDBError, KeyError):
    """Raised when an operation targets an object_id that does not exist."""

    def __init__(self, object_id: str):
        super().__init__(f"Object not found: {object_id!r}")
        self.object_id = object_id


class DuplicateObject(ObjectVectorDBError):
    """Raised when add() is called with an object_id that already exists."""

    def __init__(self, object_id: str):
        super().__init__(f"Object already exists: {object_id!r}")
        self.object_id = object_id


class VectorFieldNotRegistered(ObjectVectorDBError):
    """Raised when a vector operation targets an unregistered field."""

    def __init__(self, name: str):
        super().__init__(
            f"Vector field {name!r} is not registered. "
            f"Call register_vector_field() first, or pass auto_register=True."
        )
        self.name = name


class DimensionMismatch(ObjectVectorDBError):
    """Raised when a vector's length does not match its registered dimensionality."""

    def __init__(self, name: str, expected: int, got: int):
        super().__init__(f"Vector {name!r} expects dim={expected}, got dim={got}")
        self.name = name
        self.expected = expected
        self.got = got


class SchemaError(ObjectVectorDBError):
    """Raised for schema-level violations (bad names, type conflicts, etc.)."""


class MetricMismatch(ObjectVectorDBError):
    """Raised when a search metric conflicts with the existing index's metric."""

    def __init__(self, field: str, requested: str, index_metric: str):
        super().__init__(
            f"search() on {field!r} requested metric={requested!r}, but the existing index "
            f"uses metric={index_metric!r}. Pass metric={index_metric!r}, "
            f"or drop_index({field!r}) and rebuild with the desired metric."
        )
        self.field = field
        self.requested = requested
        self.index_metric = index_metric
