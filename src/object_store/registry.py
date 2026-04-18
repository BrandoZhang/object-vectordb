"""Schema registry stored as a JSON sidecar next to the Lance table directory.

The registry is the source of truth for:
  - which columns are vector fields (vs. regular properties)
  - the dimensionality of each vector field
  - the optional human description of a vector field
  - index configuration parameters (num_partitions, metric, etc.) — because
    lancedb's `index_stats()` does not round-trip the originally-passed params

Single-writer assumption. Atomic writes via os.replace.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REGISTRY_FILENAME = "object_store_registry.json"
REGISTRY_VERSION = 1
VECTOR_COLUMN_PREFIX = "__vec_"


@dataclass
class VectorFieldRecord:
    name: str
    dim: int
    column: str
    description: str | None = None
    index: dict[str, Any] | None = None  # {"type", "metric", "num_partitions", ...}


@dataclass
class RegistryState:
    vector_fields: dict[str, VectorFieldRecord] = field(default_factory=dict)
    property_columns: list[str] = field(default_factory=list)
    version: int = REGISTRY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "vector_fields": {k: asdict(v) for k, v in self.vector_fields.items()},
            "property_columns": sorted(set(self.property_columns)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryState:
        vfs = {
            name: VectorFieldRecord(**rec) for name, rec in data.get("vector_fields", {}).items()
        }
        return cls(
            vector_fields=vfs,
            property_columns=list(data.get("property_columns", [])),
            version=int(data.get("version", REGISTRY_VERSION)),
        )


class SchemaRegistry:
    """Lightweight on-disk registry. Reads on construction, writes on mutation."""

    def __init__(self, uri: str):
        self._uri = uri
        self._path = Path(uri) / REGISTRY_FILENAME
        self.state: RegistryState = self._load()

    def _load(self) -> RegistryState:
        if not self._path.exists():
            return RegistryState()
        with self._path.open() as f:
            return RegistryState.from_dict(json.load(f))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(self.state.to_dict(), f, indent=2, sort_keys=True)
        os.replace(tmp, self._path)

    # ---- vector fields ----

    def vector_column(self, name: str) -> str:
        return VECTOR_COLUMN_PREFIX + name

    def has_vector(self, name: str) -> bool:
        return name in self.state.vector_fields

    def get_vector(self, name: str) -> VectorFieldRecord | None:
        return self.state.vector_fields.get(name)

    def list_vectors(self) -> list[VectorFieldRecord]:
        return list(self.state.vector_fields.values())

    def add_vector(self, name: str, dim: int, description: str | None = None) -> VectorFieldRecord:
        rec = VectorFieldRecord(
            name=name,
            dim=dim,
            column=self.vector_column(name),
            description=description,
        )
        self.state.vector_fields[name] = rec
        self._save()
        return rec

    def remove_vector(self, name: str) -> None:
        self.state.vector_fields.pop(name, None)
        self._save()

    def rename_vector(self, old: str, new: str) -> VectorFieldRecord:
        rec = self.state.vector_fields.pop(old)
        rec.name = new
        rec.column = self.vector_column(new)
        self.state.vector_fields[new] = rec
        self._save()
        return rec

    def set_index(self, name: str, index: dict[str, Any] | None) -> None:
        rec = self.state.vector_fields[name]
        rec.index = index
        self._save()

    # ---- property columns ----

    def has_property(self, name: str) -> bool:
        return name in self.state.property_columns

    def add_property(self, name: str) -> None:
        if name not in self.state.property_columns:
            self.state.property_columns.append(name)
            self._save()

    def remove_property(self, name: str) -> None:
        if name in self.state.property_columns:
            self.state.property_columns.remove(name)
            self._save()

    def rename_property(self, old: str, new: str) -> None:
        if old in self.state.property_columns:
            self.state.property_columns.remove(old)
        if new not in self.state.property_columns:
            self.state.property_columns.append(new)
        self._save()

    def list_properties(self) -> list[str]:
        return sorted(self.state.property_columns)
