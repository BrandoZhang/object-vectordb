# Technical Design

This page explains *how* the library is implemented. It assumes you have
read [concepts.md](concepts.md). For *why* LanceDB is the backend (and how
this library compares to other vector DBs and abstraction layers), see
[comparison.md](comparison.md).

## Module layout

```
src/object_vectordb/
├── __init__.py      # public re-exports
├── types.py         # dataclasses: ObjectData, SearchResult, ObjectUpdate,
│                    #              VectorFieldInfo, IndexInfo
├── exceptions.py    # ObjectNotFound, DuplicateObject, VectorFieldNotRegistered,
│                    # DimensionMismatch, SchemaError, MetricMismatch,
│                    # ObjectVectorDBError
├── db.py            # ObjectVectorDB — the DB handle. Opens the LanceDB URI,
│                    #   exposes collection() / list_collections() / drop_collection().
│                    #   Uses lancedb.connect() only; no pyarrow.
├── collection.py    # Collection — per-collection API. Owns the LanceDBBackend
│                    #   and delegates all object operations. No lancedb or pyarrow.
├── backend.py       # LanceDBBackend — all LanceDB + pyarrow code. One instance
│                    #   per Collection; receives a shared lancedb connection.
├── registry.py      # SchemaRegistry (root) + CollectionRegistry (per-collection
│                    #   proxy). JSON sidecar with two-level structure.
├── fusion.py        # rrf_merge — module-level rank-fusion utility.
├── scoring.py       # Per-metric distance → similarity score conversion.
└── arrow_utils.py   # Python-value → pyarrow-type inference, record-batch helpers.
```

### Layering rule

```
ObjectVectorDB   (public, DB handle)
    │
    └── Collection   (public, per-collection operations)
            │
            ├── CollectionRegistry   (scoped view of the JSON sidecar)
            └── LanceDBBackend        (lancedb + pyarrow)
```

- `db.py` only calls `lancedb.connect(uri)`; no pyarrow.
- `collection.py` has no dependency on `lancedb` or `pyarrow`.
- All storage-engine specifics live inside `backend.py`.

To swap backends in the future (e.g. to Qdrant or Milvus), you would write a
new class with the same `LanceDBBackend` method signatures and change the
import in `collection.py`. This is **not** a formal plugin system — there is
no abstract base class and no registry of backends. Just disciplined layering.

## Schema registry

The registry is a JSON file written next to the Lance table directory:

```
<uri>/
├── <table_name>.lance/                  # LanceDB's own files
└── object_vectordb_registry.json           # our metadata sidecar
```

It tracks:

- Which columns are vector fields vs. properties.
- Each vector field's dimensionality.
- Each vector field's optional human description.
- Each vector field's index configuration (type, metric, and the originally-passed
  `num_partitions` / `num_sub_vectors` / etc. that LanceDB's `index_stats()`
  does not round-trip).

Shape (version 2 — namespaced by collection):

```json
{
  "version": 2,
  "collections": {
    "videos": {
      "vector_fields": {
        "text_openai": {
          "name": "text_openai",
          "dim": 1536,
          "column": "__vec_text_openai",
          "description": "text-embedding-3-small",
          "index": {
            "type": "IVF_PQ",
            "metric": "cosine",
            "num_partitions": 256,
            "num_sub_vectors": 16
          }
        }
      },
      "property_columns": ["title", "views", "tags"]
    },
    "images": {
      "vector_fields": {"...": "..."},
      "property_columns": ["..."]
    }
  }
}
```

Backends never see the root `SchemaRegistry` directly — they receive a
`CollectionRegistry` proxy that scopes all reads/writes to a single collection.
This keeps the backend code identical whether it's operating on collection A
or collection B.

Writes go through `os.replace(tmp, final)` for atomicity. The registry
assumes a single writer — there is no file-level lock.

### Why JSON sidecar, not a Lance metadata table?

The registry is ~1 KB of config-shaped data read on every `ObjectVectorDB`
method call. A JSON file loads in microseconds, versions cleanly in git for
test repos, is human-inspectable, and avoids creating a second Lance dataset
just to store a handful of settings. A Lance-table registry would introduce
bootstrap problems (how do you read the registry before it exists?) and
force every registry read through the LanceDB query engine, for no gain.

If multi-writer support is ever required, the registry abstraction is
localized enough that switching to a transactional Lance table at
`<uri>/_registry` is a single-file change behind `SchemaRegistry`.

## Column naming convention

Vector columns are stored with the prefix `__vec_` inside the Lance table;
property columns are stored under their own names:

| Public name       | Internal Lance column |
| ----------------- | --------------------- |
| property `title`  | `title`               |
| vector `text_openai` | `__vec_text_openai` |

The prefix is an implementation detail — the public API always uses the
unprefixed name. The registry maps between the two. Property names that
start with `__vec_` are rejected with `SchemaError` to prevent collisions.

## Zero-copy schema evolution

LanceDB's `table.add_columns(pa.field(name, dtype))` adds a column using a
**metadata-only** path when given a pyarrow `Field` (not a SQL expression).
Existing rows read back with `None` for the new column; no data is rewritten.
We use this for three cases:

1. `register_vector_field(name, dim)` — adds a
   `pa.list_(pa.float32(), list_size=dim)` column (FixedSizeList; required
   for LanceDB to index it).
2. First write of a new property — the backend infers the Arrow type from the
   sample value, calls `add_columns(pa.field(name, dtype))`, then performs
   the write.
3. Dropping: `table.drop_columns(cols)` is also metadata-only.

### Property-type inference

`arrow_utils.python_value_to_arrow_type()` maps a sample Python value to a
pyarrow type:

| Python sample               | Arrow type                          |
| --------------------------- | ----------------------------------- |
| `True` / `False`            | `bool_`                             |
| `int`                       | `int64`                             |
| `float`                     | `float64`                           |
| `str`                       | `string` (utf8)                     |
| `bytes`                     | `binary`                            |
| `list[...]`                 | `list_<elem>` (recursive inference) |
| `dict`                      | `string` (JSON-encoded on write)    |

Inference order matters: `bool` is checked before `int` because `bool` is a
subclass of `int` in Python. Inference on `None` alone is rejected with
`SchemaError` — we cannot derive a column type from a null value. Write a
non-None value first, then you can subsequently clear that column.

## LanceDB API usage (reference)

| Operation                         | Call                                                                                                                                                                     |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Connect / open / create           | `lancedb.connect(uri)` → `db.table_names()`, `db.open_table()`, `db.create_table(name, schema=...)`                                                                       |
| Zero-copy add vector column       | `table.add_columns(pa.field(col, pa.list_(pa.float32(), dim)))`                                                                                                          |
| Zero-copy add property column     | `table.add_columns(pa.field(name, inferred_type))`                                                                                                                       |
| Zero-copy drop column             | `table.drop_columns([col, ...])`                                                                                                                                         |
| Rename column                     | `table.alter_columns({"path": old, "rename": new})`                                                                                                                      |
| Upsert with column-level merge    | `table.merge_insert("object_id").when_matched_update_all().when_not_matched_insert_all().execute(batch)`                                                                  |
| Point read by id                  | `table.search().where("object_id = 'x'", prefilter=True).limit(1).to_list()`                                                                                             |
| Count rows matching a filter      | `table.count_rows(filter_sql)`                                                                                                                                           |
| Null-clear a scalar cell          | `table.update(where=..., values_sql={col: f"CAST(NULL AS {sql_type})"})`                                                                                                 |
| Null-clear a vector cell          | `merge_insert(...).when_matched_update_all().execute(batch_with_null_fixed_size_list)`                                                                                   |
| ANN search                        | `table.search(vec, vector_column_name=col).distance_type(metric).where(...).select(...).limit(n).nprobes(...).refine_factor(...).to_list()` — rows include `_distance`.  |
| Create index                      | `table.create_index(metric=..., vector_column_name=col, index_type=..., num_partitions=..., num_sub_vectors=..., replace=True)`                                           |
| List indices                      | `table.list_indices()`                                                                                                                                                   |
| Drop index                        | `table.drop_index(name)`                                                                                                                                                 |
| Read full/filtered table          | `table.search().where(...).select(...).limit(N).to_arrow()`                                                                                                               |

### Gotchas we encode in the backend

- **No auto-column inference on `add`.** LanceDB does not grow the schema
  from `table.add(rows)`. We always pre-ensure every column exists via
  `add_columns` before inserting.
- **No primary-key uniqueness.** LanceDB will happily duplicate `object_id`
  rows. `add()` does a pre-check with `count_rows("object_id = 'x'")` and
  raises `DuplicateObject` if the id exists. This is race-prone in a
  multi-writer world — documented as single-writer.
- **`update` is silent on empty match.** `table.update(where=...)` affects 0
  rows without complaint. `ObjectVectorDB.update()` pre-checks existence and
  raises `ObjectNotFound`.
- **`values={col: None}` has round-trip bugs** (LanceDB issues #1325, #3105).
  We avoid this path entirely. Scalar null-clearing uses
  `values_sql={col: "CAST(NULL AS STRING)"}` with an explicit typed cast;
  vector null-clearing goes through `merge_insert` with a FixedSizeList
  null batch (Arrow-level nulls round-trip cleanly on that path).
- **Index metric wins at search time.** Once an index exists, LanceDB ignores
  the `distance_type()` the caller passed. We read the stored metric from the
  registry and raise `MetricMismatch` if the caller's explicit `metric=`
  disagrees.
- **Renaming an indexed vector column orphans the index.** LanceDB's
  `alter_columns` renames the column in the schema but does not update the
  index's column reference. `rename_field()` therefore drops the index,
  renames, then recreates the index using the stored config.
- **`batch_update` rejects duplicate ids inside a single batch.** LanceDB's
  `merge_insert` against multiple source rows that share a key is
  implementation-defined, and our column-signature grouping would split such
  rows into separate groups with unreliable apply-order. `batch_update`
  therefore raises `DuplicateObject` on an intra-batch duplicate, mirroring
  `add_many`'s behavior.

## Score conversion

LanceDB returns a `_distance` column on search results where *lower* is
better. The public API returns a similarity `score` where *higher* is better.
The conversion is per-metric:

| `metric`  | LanceDB `_distance`       | `score = f(_distance)`     | Range       | Notes                                                              |
| --------- | ------------------------- | -------------------------- | ----------- | ------------------------------------------------------------------ |
| `cosine`  | `1 − cosine_similarity`   | `1 − distance`             | `[-1, 1]`   | Exact; recovers cosine similarity.                                 |
| `l2`      | `‖q − v‖²` (squared)      | `1 / (1 + distance)`       | `(0, 1]`    | Monotonic only. Magnitudes are not physically meaningful. Do not compare across different query vectors. |
| `dot`     | `1 − dot(q, v)`           | `1 − distance`             | unbounded   | Equals raw dot product. Verified empirically (LanceDB 0.30.x).     |

The test suite includes a calibration test per metric
(`test_search.py::test_search_dot_score_recovers_raw_dot`, etc.) that inserts
vectors with known similarities and asserts the returned score matches.

## Null-clearing strategy

A caller clears a field by passing `None`:

```python
store.update("x", properties={"title": None})
store.update("x", vectors={"image_clip": None})
```

- **Scalar property.** Looked up column type from `table.schema`, mapped to a
  SQL type via `arrow_type_to_sql_type()`, issued as
  `table.update(where="object_id = 'x'", values_sql={col: "CAST(NULL AS BIGINT)"})`.
  The SQL cast path is reliable; the `values={col: None}` path has known bugs.
- **Vector.** Built a pyarrow RecordBatch with one row — the `object_id`
  string and the vector column typed as
  `pa.list_(pa.float32(), list_size=dim)` with a single null entry. Sent
  via `merge_insert("object_id").when_matched_update_all().execute(batch)`.
  Arrow-level nulls on FixedSizeList columns round-trip cleanly.

## `add`, `update`, and `batch_update` write paths

- `add(object_id, properties, vectors)`
  1. Pre-check: `count_rows("object_id = 'x'") > 0` → `DuplicateObject`.
  2. Ensure every property column exists (auto-add via `add_columns`).
  3. Build a single-row pyarrow Table with exactly the touched columns.
  4. `merge_insert("object_id").when_not_matched_insert_all().execute(batch)`.

- `update(object_id, properties?, vectors?, on_missing="raise")`
  1. If `on_missing != "insert"`: pre-check existence; on missing, raise
     `ObjectNotFound` (`"raise"`) or silently return (`"skip"`).
  2. Separate None values (clears) from non-None writes.
  3. If any writes: build a one-row batch with only the touched columns and
     run `merge_insert.when_matched_update_all()`. With `on_missing="insert"`
     the merge also adds `when_not_matched_insert_all()` (upsert). Because
     the batch only contains the touched columns, `when_matched_update_all()`
     leaves all other columns untouched.
  4. For each None clear: issue `values_sql` (scalars) or `merge_insert`
     with null-vector batch (vectors). Null-clears use update-only
     merge_insert and silently no-op on missing rows.

- `batch_update([ObjectUpdate, ...], on_missing="raise")`
  1. Reject duplicate `object_id`s inside the batch (`DuplicateObject`).
  2. If `on_missing != "insert"`: pre-check existence per id; on missing,
     either raise `ObjectNotFound` or drop the row from the batch
     (`"skip"`).
  3. Collect non-None writes into rows; collect None clears separately.
  4. **Group rows by column signature.** Within a group, every row touches
     the same column set, so `when_matched_update_all()` preserves every
     other column for every row. Without grouping, a row that specifies
     only `{"n": 1}` would land in a batch whose schema also includes `v`
     (because a sibling row set `v`), with `v=None` — and
     `when_matched_update_all()` would null-out the original `v` for that
     row. Grouping avoids that failure mode.
  5. For each group: one `merge_insert` call. With `on_missing="insert"`,
     the call adds `when_not_matched_insert_all()` so missing rows become
     partial inserts.
  6. Issue per-row clears as above.

## Search path

1. Validate: `vector_field` must be registered; `query_vector` must be non-
   empty and match `dim`; requested `metric` must not conflict with the
   index's `metric` if one exists.
2. Build:
   `table.search(query, vector_column_name=col).distance_type(metric)`
   with optional `.where(...)`, `.select(...)`, `.limit(n)`,
   `.nprobes(n)`, `.refine_factor(n)`.
3. Call `to_list()` to materialize rows.
4. For each row: pop `_distance`, pop `_rowid` / `_score` if present, drop
   `__vec_*` columns from the properties dict, optionally filter to the
   caller's `select` list, compute `score = distance_to_score(distance, metric)`,
   wrap in `SearchResult`.

### Radius search (`search_within`)

`search_within(query, field, max_distance, min_distance=..., exact=...)`
reuses the same validation / metric-resolution / row-materialization helpers
(`_prepare_vector_query`, `_row_to_result`) as `search`. The difference is the
builder chain: it calls `builder.distance_range(lower, upper)` so filtering
happens **inside LanceDB** — no Python post-filter, no over-fetch. The lower
bound is passed as `None` (not `0.0`) when the caller omits `min_distance`
so that negative distances produced by `metric="dot"` are not silently
excluded. `limit=None` becomes a sentinel `2**31 - 1` because the LanceDB
builder requires a limit.

Radius queries against an IVF index are approximate (matches in unprobed
partitions are missed); `exact=True` calls `builder.bypass_vector_index()` to
force a full scan when completeness matters more than latency. `search`
tolerates approximation naturally (it still returns `k` results); radius
queries do not — hence the first-class `exact` knob on `search_within` only.

## RRF (Reciprocal Rank Fusion)

`rrf_merge(list1, list2, ..., k=60, limit=None)` (from `object_vectordb`) is a pure Python
utility that combines multiple `SearchResult` lists (e.g. a text-vector
search and an image-vector search) into a single fused ranking:

```
fused_score(id) = sum over input lists of  1 / (k + rank_in_that_list(id))
```

Objects not present in a given list contribute 0 for that list. Ties break by
first-seen order across the concatenated inputs. Properties from the first
occurrence of each id are preserved on the returned `SearchResult`.

## Concurrency model

Single-writer only. The JSON sidecar is not locked. Concurrent readers are
safe: LanceDB supports concurrent reads on a table, and the registry is
re-read on every ObjectVectorDB construction. For multi-writer, route writes
through a queue consumer running in a single process.

### Side effects under the single-writer contract

These are consequences of the single-writer assumption that a caller should
know about before violating it:

- **TOCTOU on `add()` / `update()` pre-checks.** `add()` does
  `count_rows → merge_insert.when_not_matched_insert_all`; `update()` and
  `batch_update()` do `exists → merge_insert`. A concurrent writer between
  the check and the merge_insert can cause:
  - Two `add()` calls with the same id: the loser's `when_not_matched_insert_all`
    silently no-ops — **no error is raised**, the caller believes the write
    succeeded.
  - Concurrent `delete` racing an `update()` with the default
    `on_missing="raise"`: the merge_insert is update-only, so it silently
    no-ops and the row stays deleted. The caller sees `update()` return
    successfully even though no row was written. `batch_update` is the
    same. Pass `on_missing="insert"` if you want the upsert behavior
    instead — that path **silently re-inserts a partial row** containing
    only the columns the update touched; every other column is null. Both
    behaviors are pinned by tests in `tests/test_update.py`
    (`test_update_default_silently_noops_when_concurrently_deleted` and
    `test_update_on_missing_insert_resurrects_partial_row`).
- **Registry sidecar.** Two processes calling `register_vector_field` or
  writing schema concurrently can lose one update: each reads the JSON, each
  writes its own modified copy via `os.replace`. The Lance columns exist, but
  the registry forgets them — subsequent opens see a "property column
  starting with `__vec_`" state that property-name validation rejects.
- **`batch_update` is not atomic across groups.** Rows are split into
  column-signature groups, each executed as its own `merge_insert`, followed
  by per-row null-clears. If the process crashes mid-batch, the batch is
  partially applied. Callers that need atomicity should either keep batches
  small enough to recover by re-running or integrate with LanceDB's
  versioning (`table.checkout_latest` / restore).
- **Search sees an indexed delta.** Vectors added or updated after an IVF
  index was built are searched via LanceDB's unindexed-delta path until the
  next `optimize()` / index refresh. Results are correct, but latency and
  recall can shift once the delta grows. Call `Collection.optimize()`
  periodically after large ingests.
- **`register_vector_field` dim-mismatch recovery.** Registering with a
  different `dim` after the first registration raises `DimensionMismatch`.
  If the first registration used the wrong dimension and rows already
  reference it, the only recovery is `drop_fields([name])` followed by
  `register_vector_field(name, correct_dim)` and re-ingest. The registry
  is the source of truth for `dim`; it cannot be changed in place.
