# Changelog

## 0.2.0 (2026-04-24)

### Breaking changes

- **`on_missing` removed from `update()` and `batch_update()`.**
  `update()` now always raises `ObjectNotFound` when the target id is absent;
  there is no silent skip or implicit insert path.
  Use `upsert()` for insert-if-missing semantics.

- **Registry format: JSON sidecar → Arrow field metadata.**
  Vector field records (dim, description, index config) are now stored in the
  Arrow field metadata of each `__vec_<name>` column inside the Lance manifest.
  The old `object_vectordb_registry.json` sidecar is auto-migrated on first open
  of an existing URI and then deleted.  There is no manual migration step.

- **`OnMissing` type removed** from `object_vectordb.types` and from the public
  `__all__` export.

### New features

- **`storage_options` passthrough on `ObjectVectorDB(uri=…, storage_options=…)`.**
  Forwarded to `lancedb.connect`.  Required for multi-writer deployments
  against S3 / object stores so Lance can coordinate manifest commits
  (conditional PUT, external commit lock, etc.).  Without it, concurrent
  writers to the same object-store URI can silently clobber each other's
  manifest commits.  See docs/concurrency.md.

- **Multi-writer support.**
  - `add()` and `batch_add()` use `MergeResult.num_inserted_rows` instead of a
    pre-check `count_rows` query to detect duplicate ids.  No TOCTOU window.
  - `update()` uses `MergeResult.num_updated_rows` (and `UpdateResult.rows_updated`
    for null-scalar clears) to detect missing rows.  No TOCTOU window.
  - `batch_update()` performs a single batch existence check (one `IN (...)` query
    for all ids) instead of N per-row `exists()` calls.
  - Schema mutations (`register_vector_field`, `set_index`, `drop_fields`,
    `rename_field`, `create_index`) use `_with_retry` to handle Lance manifest
    commit conflicts with bounded exponential backoff (5 attempts, 50–800 ms).
  - Concurrent `add_columns` calls for the same column name are idempotent
    ("column already exists" errors are swallowed).
  - Concurrent `create_table` for the same collection name no longer
    races — one writer creates, the rest fall through to `open_table`.
  - Concurrent `register_vector_field` (or auto-register) with **different
    dims** now detects the mismatch after the "already exists" race and
    raises `DimensionMismatch`, rather than writing conflicting dim
    metadata over the actual column type.
  - `batch_update` now checks `MergeResult.num_updated_rows` per
    column-signature group (and `UpdateResult.rows_updated` on null-clear
    ops) to catch rows deleted concurrently between the pre-check and the
    write.  Previously the batch silently skipped those rows.
  - `_ensure_object_id_index` is now best-effort on existing tables:
    read-only readers (or concurrent writers racing on first open) no
    longer crash.  On newly-created tables it is still required.

- **`upsert()` remains the only insert-if-missing path.**
  Callers that previously used `update(..., on_missing="insert")` should switch to
  `upsert()`.  Merge semantics are unchanged: existing fields not in the call are
  preserved; missing rows are inserted as partial rows.

### Notes

- **Same-id concurrent inserts are still best-effort.**
  Lance treats no-match `merge_insert` as a commutative append, so two writers
  that both observe "no row" at read time will both commit, producing duplicate
  rows.  `add()` raises `DuplicateObject` whenever the row was visible at read
  time, closing the old `exists()`→write TOCTOU, but it does not prevent two
  racing inserts from both landing.  Use an external lock when strict same-id
  uniqueness is required under concurrency.  Writes to distinct ids, and
  updates/upserts against the same *existing* row, are fully safe.

- **`batch_update` is still not atomic across column-signature groups.**
  A `batch_update` whose rows have differing column sets executes as N independent
  `merge_insert` calls.  Each call is atomic and conflict-retried; the batch as a
  whole is not.  For full atomicity, issue homogeneous batches (same columns for
  every row).

- **`lancedb` minimum version bumped to `0.30.0`** (`MergeResult` with named fields
  is required).
