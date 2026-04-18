"""Schema registry stored as a JSON sidecar next to the Lance table directory.

The registry is the source of truth for:
  - which collections exist at a given URI
  - which columns are vector fields (vs. regular properties) within each collection
  - the dimensionality of each vector field
  - the optional human description of a vector field
  - index configuration parameters (num_partitions, metric, etc.) — because
    lancedb's `index_stats()` does not round-trip the originally-passed params

Two-level shape: `{version, collections: {name: {vector_fields, property_columns}}}`.
A `CollectionRegistry` proxy scopes reads/writes to a single collection; backends
receive the proxy so they never see the cross-collection state.

Single-writer assumption. Atomic writes via os.replace.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REGISTRY_FILENAME = "object_vectordb_registry.json"
REGISTRY_VERSION = 2
VECTOR_COLUMN_PREFIX = "__vec_"


@dataclass
class VectorFieldRecord:
    name: str
    dim: int
    column: str
    description: str | None = None
    index: dict[str, Any] | None = None


@dataclass
class CollectionState:
    vector_fields: dict[str, VectorFieldRecord] = field(default_factory=dict)
    property_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_fields": {k: asdict(v) for k, v in self.vector_fields.items()},
            "property_columns": sorted(set(self.property_columns)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollectionState:
        vfs = {
            name: VectorFieldRecord(**rec) for name, rec in data.get("vector_fields", {}).items()
        }
        return cls(
            vector_fields=vfs,
            property_columns=list(data.get("property_columns", [])),
        )


@dataclass
class RegistryState:
    collections: dict[str, CollectionState] = field(default_factory=dict)
    version: int = REGISTRY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "collections": {k: v.to_dict() for k, v in self.collections.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryState:
        version = int(data.get("version", 0))
        if version != REGISTRY_VERSION:
            log.warning(
                "Registry version %s not understood (expected %s); starting empty.",
                version,
                REGISTRY_VERSION,
            )
            return cls()
        collections = {
            name: CollectionState.from_dict(c) for name, c in data.get("collections", {}).items()
        }
        return cls(collections=collections, version=version)


class SchemaRegistry:
    """Root registry. Owns the on-disk JSON sidecar and hands out per-collection views."""

    def __init__(self, uri: str):
        self._uri = uri
        self._path = Path(uri) / REGISTRY_FILENAME
        self.state: RegistryState = self._load()

    def _load(self) -> RegistryState:
        if not self._path.exists():
            return RegistryState()
        try:
            with self._path.open() as f:
                return RegistryState.from_dict(json.load(f))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("Registry at %s unreadable (%s); starting empty.", self._path, exc)
            return RegistryState()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(self.state.to_dict(), f, indent=2, sort_keys=True)
        os.replace(tmp, self._path)

    # ---- collection-level operations ----

    def collection(self, name: str) -> CollectionRegistry:
        if name not in self.state.collections:
            self.state.collections[name] = CollectionState()
            self._save()
        return CollectionRegistry(self, name)

    def list_collections(self) -> list[str]:
        return sorted(self.state.collections.keys())

    def has_collection(self, name: str) -> bool:
        return name in self.state.collections

    def drop_collection(self, name: str) -> None:
        if name in self.state.collections:
            del self.state.collections[name]
            self._save()


class CollectionRegistry:
    """Per-collection view over the parent SchemaRegistry.

    Backends receive this proxy; they never see other collections' state.
    Writes are proxied to the parent, which persists the whole file.
    """

    def __init__(self, root: SchemaRegistry, collection_name: str):
        self._root = root
        self._name = collection_name

    @property
    def _state(self) -> CollectionState:
        return self._root.state.collections[self._name]

    def _save(self) -> None:
        self._root._save()

    def vector_column(self, name: str) -> str:
        return VECTOR_COLUMN_PREFIX + name

    def has_vector(self, name: str) -> bool:
        return name in self._state.vector_fields

    def get_vector(self, name: str) -> VectorFieldRecord | None:
        return self._state.vector_fields.get(name)

    def list_vectors(self) -> list[VectorFieldRecord]:
        return list(self._state.vector_fields.values())

    def add_vector(self, name: str, dim: int, description: str | None = None) -> VectorFieldRecord:
        rec = VectorFieldRecord(
            name=name,
            dim=dim,
            column=self.vector_column(name),
            description=description,
        )
        self._state.vector_fields[name] = rec
        self._save()
        return rec

    def remove_vector(self, name: str) -> None:
        self._state.vector_fields.pop(name, None)
        self._save()

    def rename_vector(self, old: str, new: str) -> VectorFieldRecord:
        rec = self._state.vector_fields.pop(old)
        rec.name = new
        rec.column = self.vector_column(new)
        self._state.vector_fields[new] = rec
        self._save()
        return rec

    def set_index(self, name: str, index: dict[str, Any] | None) -> None:
        rec = self._state.vector_fields[name]
        rec.index = index
        self._save()

    def has_property(self, name: str) -> bool:
        return name in self._state.property_columns

    def add_property(self, name: str) -> None:
        if name not in self._state.property_columns:
            self._state.property_columns.append(name)
            self._save()

    def remove_property(self, name: str) -> None:
        if name in self._state.property_columns:
            self._state.property_columns.remove(name)
            self._save()

    def rename_property(self, old: str, new: str) -> None:
        cols = self._state.property_columns
        if old in cols:
            cols.remove(old)
        if new not in cols:
            cols.append(new)
        self._save()

    def list_properties(self) -> list[str]:
        return sorted(self._state.property_columns)
