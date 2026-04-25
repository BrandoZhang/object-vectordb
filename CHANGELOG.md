# Changelog

Versioning follows [SemVer 2.0](https://semver.org/spec/v2.0.0.html). Pre-release
artifacts produced from `main` use the PEP 440 `0.X.Y.devN` form (Python's
nearest equivalent to a SemVer pre-release).

## 0.1.0 (2026-04-25) — initial release

First tagged release. The library exposes an object-centric API
(`ObjectVectorDB` → `Collection` → `LanceDBBackend`) with multi-named
vector fields per object, dynamic property schema, zero-copy schema
evolution on the Lance manifest, and per-metric similarity scoring.

### Features

- **Public surface**: `ObjectVectorDB`, `Collection`, dataclasses
  (`ObjectAdd`, `ObjectData`, `ObjectUpdate`, `SearchResult`,
  `VectorFieldInfo`, `IndexInfo`), `rrf_merge`, and the typed exception
  hierarchy (`ObjectVectorDBError` and friends).
- **Per-collection schema isolation.** Two collections at the same URI can
  register the same vector field at different dimensionalities without
  colliding.
- **Object lifecycle**: `add` / `batch_add` / `get` / `batch_get` / `exists`
  / `delete`, plus `update` / `batch_update` (merge semantics, raise on
  missing) and `upsert` (insert-if-missing with merge semantics).
- **Schema evolution**: `register_vector_field`, `drop_fields`,
  `rename_field`, `schema()`. All zero-copy via Lance manifest changes;
  registry state is encoded in Arrow field metadata
  (`ovdb_dim` / `ovdb_description` / `ovdb_index`) on the
  `__vec_<name>` columns plus a `ovdb_schema_version` sentinel on
  `object_id`. No JSON sidecar.
- **Search**: `search` (top-k ANN), `search_within` (radius / distance-bounded),
  `export_vectors` (bulk read), `list_objects` (filter + paginate).
  Per-metric score conversion (`cosine` / `l2` / `dot`); `MetricMismatch`
  raised when caller's metric conflicts with an existing index.
- **Index management**: `create_index` / `rebuild_index` / `drop_index` /
  `index_info`. Auto-created `BTREE` scalar index on `object_id` so point
  lookups are O(log N) from the first write.
- **Multi-writer support.** Schema mutations retry on Lance manifest
  commit conflicts via `_with_retry` (5 attempts, 50–800 ms backoff).
  `add` / `update` / `batch_update` use Lance's `MergeResult` row counts
  to detect duplicates and missing rows without TOCTOU windows.
  Distinct-id writes and updates/upserts against existing rows are
  multi-writer safe; same-id concurrent inserts can still produce
  duplicates because Lance treats unmatched `merge_insert` as commutative
  append — see `docs/concurrency.md` for the full multi-writer story and
  the `storage_options` knob for object-store deployments.
- **Hybrid retrieval helper**: `rrf_merge` for fusing multiple
  `SearchResult` lists (e.g., one text-vector, one image-vector).

### Documentation

`docs/` covers concepts, architecture, full API reference, filter
syntax, comparison with other vector DBs, concurrency, and testing.
Performance baselines are in the README's Performance section, with
quick-tier benchmarks at dim=1024 recorded against post-BTREE numbers.

### License

Apache 2.0.
