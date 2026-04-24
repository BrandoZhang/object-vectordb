# API Reference

All public names are re-exported from the top-level `object_vectordb` package.

```python
from object_vectordb import (
    ObjectVectorDB, Collection, rrf_merge,
    ObjectData, ObjectUpdate, SearchResult, VectorFieldInfo, IndexInfo,
    ObjectVectorDBError,
    ObjectNotFound, DuplicateObject, VectorFieldNotRegistered,
    DimensionMismatch, SchemaError, MetricMismatch,
)
```

The API deliberately uses only Python-native types (`str`, `int`, `float`,
`list`, `dict`, `numpy.ndarray`). No `lancedb` or `pyarrow` types leak out.

## Class: `ObjectVectorDB` (DB handle)

### Constructor

```python
ObjectVectorDB(uri: str, storage_options: dict[str, str] | None = None)
```

- `uri` — local directory path or a cloud URI (`s3://…`) that LanceDB can
  open. Collection metadata is stored in the Lance manifest; there is no
  on-disk sidecar state.
- `storage_options` — forwarded to `lancedb.connect`. Required on S3 (and
  other object stores) with multiple concurrent writers, so Lance can
  coordinate manifest commits; see `docs/concurrency.md`.

Construction is cheap: it opens the LanceDB directory. No collections are
created; call `collection()` to open/create one.

---

### `collection`

```python
collection(name: str, auto_register: bool = False) -> Collection
```

Open or create a named collection. If the underlying Lance table doesn't
exist, it's created with just the `object_id` column; vector and property
columns are added later via `register_vector_field` and the first write.

- `auto_register` — per-call flag (not persisted). When `True`, writing a
  vector under a previously-unseen name on the returned collection
  implicitly registers it with `dim = len(vector)`. Default `False`.
- Raises `ValueError` if `name` is empty.

---

### `list_collections`

```python
list_collections() -> list[str]
```

Collection names registered at this URI, intersected with the Lance tables
actually present on disk. Sorted alphabetically.

---

### `has_collection`

```python
has_collection(name: str) -> bool
```

`True` iff the Lance table and the registry entry both exist.

---

### `drop_collection`

```python
drop_collection(name: str) -> None
```

Delete the Lance table and its registry entry. Silent no-op if the
collection doesn't exist.

---

## Class: `Collection` (per-collection API)

Obtained from `ObjectVectorDB.collection(name)`. All per-object operations
live here. Each collection owns its own vector-field registry, property
columns, and Lance table.

### `register_vector_field`

```python
register_vector_field(name: str, dim: int, description: str | None = None) -> VectorFieldInfo
```

Declares a vector field. Adds the underlying column via LanceDB's zero-copy
metadata-only path; existing rows read back with `None` for the new field.

- Idempotent if `(name, dim)` matches an existing registration.
- Raises `DimensionMismatch` if `name` is already registered with a
  different dim.
- Raises `SchemaError` if `name` is empty or starts with the reserved
  prefix `__vec_`.

Returns a `VectorFieldInfo` describing the registered field.

---

### `list_vector_fields`

```python
list_vector_fields() -> list[VectorFieldInfo]
```

Lists all registered vector fields and their current state (dim, whether an
index exists, description).

---

### `add`

```python
add(
    object_id: str,
    properties: dict[str, Any] | None = None,
    vectors: dict[str, list[float] | None] | None = None,
) -> None
```

Inserts a new object. Raises `DuplicateObject` if `object_id` already exists.

- Every property key not yet in the schema is auto-added by inferring the
  Arrow type from its value (see
  [architecture.md § property-type inference](architecture.md#property-type-inference)).
- Every vector field referenced must be registered (or `auto_register=True`
  at construction). Dimension is validated; `DimensionMismatch` on mismatch.
- Passing `None` for a property value is only allowed if the column already
  exists; otherwise raises `SchemaError` (we cannot infer a type from
  `None`).

---

### `batch_add`

```python
batch_add(items: Iterable[ObjectAdd]) -> None
```

Bulk insert. Each item is an [`ObjectAdd`](#objectadd) with fields
`object_id`, `properties?`, `vectors?` — same shape and validation as
`add()`. Raises `DuplicateObject` if any id already exists in the table
or appears twice in the batch.

---

### `get`

```python
get(object_id: str) -> ObjectData | None
```

Returns the full object including all properties and all registered vectors
(vectors not set on this object return as `None`). Returns `None` if the
object does not exist.

---

### `batch_get`

```python
batch_get(object_ids: Iterable[str]) -> list[ObjectData | None]
```

Fetch multiple objects in a single scan. Returns a list aligned to the
input order; each position is either the matching `ObjectData` or `None`
if that id is absent. Duplicate input ids return the same object at each
position. Empty input returns `[]`. Much cheaper than N calls to `get()`
because it runs one `where object_id IN (...)` scan instead of N.

---

### `exists`

```python
exists(object_id: str) -> bool
```

Fast point-existence check via `count_rows`.

---

### `delete`

```python
delete(object_id: str) -> None
```

Silent no-op if the id does not exist. Otherwise deletes the row.

---

### `update`

```python
update(
    object_id: str,
    properties: dict[str, Any] | None = None,
    vectors: dict[str, list[float] | None] | None = None,
) -> None
```

Merge-update: only the specified fields are touched.

- Non-None values overwrite the existing value.
- `None` **clears** the field on this specific object (the column stays,
  the cell becomes null).
- New properties introduced here are auto-added to the schema.
- Raises `ObjectNotFound` if `object_id` is not in the table. Use
  `upsert()` for insert-if-missing semantics.

---

### `upsert`

```python
upsert(
    object_id: str,
    properties: dict[str, Any] | None = None,
    vectors: dict[str, list[float] | None] | None = None,
) -> None
```

Insert if missing, merge-update if present. The only insert-if-missing
path — `update()` always raises `ObjectNotFound` on absent rows.
Semantics are **merge, not replace**: unspecified fields on an existing row
are preserved, and missing rows are created as partial rows containing only
the fields you passed.

---

### `batch_update`

```python
batch_update(
    updates: Iterable[ObjectUpdate],
) -> None
```

Applies many `ObjectUpdate` records efficiently. Internally groups updates
by the set of touched columns so that `merge_insert.when_matched_update_all()`
never nulls out a column that a sibling update did not touch.

- Raises `DuplicateObject` if the same `object_id` appears twice in one
  batch (LanceDB's behavior on duplicate merge keys is undefined; we reject
  to avoid surprises).
- A single batch existence check (one `IN (...)` query) runs before any
  write; if any id is missing, `ObjectNotFound` is raised and nothing is
  written. Per-group `num_updated_rows` is also inspected after each
  merge to catch rows deleted concurrently between pre-check and write.
- For insert-if-missing semantics across a batch, issue individual
  `upsert()` calls instead.

---

### `drop_fields`

```python
drop_fields(names: Iterable[str]) -> None
```

Removes columns (properties and/or vector fields) from the schema
zero-copy. For a vector field with an index, the index is dropped first.
Unknown names are silently skipped.

---

### `rename_field`

```python
rename_field(old: str, new: str) -> None
```

Renames a property column or a vector field. For a vector field, the
internal column is also renamed (`__vec_<old>` → `__vec_<new>`); if the
field had an index, the index is dropped and recreated using the stored
configuration. Raises `SchemaError` if `new` already exists or if either
name is reserved / empty.

---

### `schema`

```python
schema() -> dict[str, dict[str, Any]]
```

Returns:

```python
{
    "properties": {"title": "string", "views": "int64", "tags": "list<item: string>"},
    "vectors":    {"text_openai": {"dim": 1536, "description": "..."},
                   "image_clip":  {"dim": 512}}
}
```

Property types are stringified Arrow types.

---

### `search`

```python
search(
    query_vector: list[float] | np.ndarray,
    vector_field: str,
    limit: int = 10,
    metric: str | None = None,
    where: str | None = None,
    select: list[str] | None = None,
    nprobes: int | None = None,
    refine_factor: int | None = None,
) -> list[SearchResult]
```

ANN search on the specified vector field.

- `query_vector` must have length equal to the registered dim. `numpy`
  arrays are coerced to `float32`.
- `metric` defaults to `"cosine"`. Allowed: `"cosine"`, `"l2"`
  (alias `"euclidean"`), `"dot"`.
- If a field has an index and `metric` disagrees with the index's metric,
  raises `MetricMismatch`. (LanceDB silently uses the index's metric in
  this case; we surface it as an error so callers don't get confusing
  scores.)
- `where` is a DataFusion SQL expression (see [filters.md](filters.md)).
- `select` restricts which properties are returned on each hit; internal
  vector columns are always stripped.
- `nprobes` / `refine_factor` pass through to LanceDB's IVF/PQ search.
- Result score is a similarity (higher = better); see
  [architecture.md § score conversion](architecture.md#score-conversion)
  for the per-metric formula.

Raises:

- `ValueError` if `query_vector` is empty.
- `VectorFieldNotRegistered` if `vector_field` is unknown.
- `DimensionMismatch` if `len(query_vector) != dim`.
- `MetricMismatch` on an index/metric conflict.

---

### `search_within`

```python
search_within(
    query_vector: list[float] | np.ndarray,
    vector_field: str,
    max_distance: float,
    *,
    min_distance: float | None = None,
    limit: int | None = None,
    metric: str | None = None,
    where: str | None = None,
    select: list[str] | None = None,
    nprobes: int | None = None,
    refine_factor: int | None = None,
    exact: bool = False,
) -> list[SearchResult]
```

Radius (distance-bounded) vector search. Returns every object whose
distance from `query_vector` lies in the half-open interval
`[min_distance or 0.0, max_distance)`. Results are sorted by ascending
distance (= descending `SearchResult.score`), matching `search()`.

- `max_distance` is in LanceDB's **native distance space** for the resolved
  metric (not the similarity score). Conversion table:

  | metric   | distance formula          | `max_distance` for `min_score` `s`    |
  | -------- | ------------------------- | ------------------------------------- |
  | `cosine` | `1 - cos_sim`             | `1 - s`                               |
  | `l2`     | squared Euclidean         | `(1 / s) - 1` (since `score = 1/(1+d)`) |
  | `dot`    | `1 - dot(q, v)`           | `1 - s` (score equals raw dot)        |

  `dot` distances can be negative (when `dot > 1`); `min_distance` must also
  be negative in that case. `cosine` distances lie in `[0, 2]`; `l2` in
  `[0, ∞)`.
- `limit=None` means unbounded. Other filter/projection args behave exactly
  as in `search()`.
- `exact=True` bypasses the ANN index for this query so the scan is
  guaranteed to find every match. **Radius queries against an IVF index are
  approximate** — matches in unprobed partitions are silently missed. Pass
  `exact=True` when completeness matters more than latency.
- Composes with `rrf_merge` unchanged (results are still rank-ordered).

Raises:

- `ValueError` if `query_vector` is empty, `max_distance` is `NaN`/`±inf`,
  `min_distance` is `NaN`/`±inf`, or `min_distance >= max_distance`.
- `VectorFieldNotRegistered` if `vector_field` is unknown.
- `DimensionMismatch` if `len(query_vector) != dim`.
- `MetricMismatch` on an index/metric conflict.

---

## Module-level: `rrf_merge`

```python
from object_vectordb import rrf_merge

rrf_merge(
    *result_lists: list[SearchResult],
    k: int = 60,
    limit: int | None = None,
) -> list[SearchResult]
```

Reciprocal Rank Fusion. Pure-Python utility that combines multiple
`SearchResult` lists (e.g. one from a text-vector search, another from an
image-vector search) into one fused ranking. Object properties on the
returned list come from the first occurrence of each id. See
[architecture.md § RRF](architecture.md#rrf-reciprocal-rank-fusion) for
the formula.

Not a method on any class — takes `SearchResult` lists and returns a new
`SearchResult` list; no backend involvement.

---

### `create_index`

```python
create_index(
    vector_field: str,
    index_type: str = "IVF_PQ",
    metric: str = "cosine",
    replace: bool = True,
    **params: Any,
) -> None
```

Creates (or replaces) an ANN index on a vector field. Common params:

- `num_partitions` — IVF coarse-cluster count.
- `num_sub_vectors` — PQ sub-vector count.

Other `**params` are passed through to LanceDB's `create_index`. The
configuration (including every `**params` entry) is stored in the registry
so `index_info()` can round-trip it — LanceDB's `index_stats()` does not.

---

### `rebuild_index`

```python
rebuild_index(vector_field: str) -> None
```

Drops and recreates the index using the stored configuration. Useful after
significant data churn (updates/deletes) that leaves the index stale.

Raises `SchemaError` if no index has ever been created on this field.

---

### `drop_index`

```python
drop_index(vector_field: str) -> None
```

Drops the index; the vector data is preserved. `search()` falls back to
brute-force (no error, just slower).

---

### `index_info`

```python
index_info(vector_field: str) -> IndexInfo | None
```

Returns the current index configuration, or `None` if no index exists.

---

### `export_vectors`

```python
export_vectors(
    vector_field: str,
    where: str | None = None,
) -> tuple[list[str], np.ndarray]
```

Bulk-reads a single vector field into memory, optionally filtered by a
DataFusion SQL `where` clause. Rows whose vector is `None` are **skipped**
so `len(ids) == embeddings.shape[0]`.

Returns `(ids, embeddings)` where `embeddings` is an `(N, dim)` float32
array.

---

### `list_objects`

```python
list_objects(
    where: str | None = None,
    select: list[str] | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> list[ObjectData]
```

Bulk-read objects matching a filter. Used for offline analysis and
pagination. Unlike `search()` this takes no query vector. If `select` is
given, only those properties are included on each `ObjectData`.

---

### `optimize`

```python
optimize() -> None
```

Calls LanceDB's `table.optimize()` — file compaction, pruning of old
versions, and incremental index updates for newly-added rows. Not called
implicitly on writes; invoke periodically after large batches (e.g. every
~100k rows or ~20 major operations).

---

## Dataclasses

### `ObjectData`

Returned by `get()` and `list()`.

```python
@dataclass
class ObjectData:
    object_id: str
    properties: dict[str, Any]
    vectors: dict[str, list[float] | None]
```

`vectors` contains an entry for every registered vector field; the value is
`None` if that vector is not set on this object.

### `SearchResult`

Returned by `search()` and `rrf_merge()`.

```python
@dataclass
class SearchResult:
    object_id: str
    score: float          # higher = more similar
    properties: dict[str, Any]
```

See [architecture.md § score conversion](architecture.md#score-conversion)
for how `score` is derived per metric.

### `ObjectAdd`

Input to `batch_add()`. Fields mirror the arguments of `add()`.

```python
@dataclass
class ObjectAdd:
    object_id: str
    properties: dict[str, Any] | None = None
    vectors: dict[str, list[float] | None] | None = None
```

### `ObjectUpdate`

Input to `batch_update()`.

```python
@dataclass
class ObjectUpdate:
    object_id: str
    properties: dict[str, Any] | None = None
    vectors: dict[str, list[float] | None] | None = None
```

### `VectorFieldInfo`

Returned by `list_vector_fields()` and `register_vector_field()`.

```python
@dataclass
class VectorFieldInfo:
    name: str
    dim: int
    has_index: bool
    description: str | None = None
```

### `IndexInfo`

Returned by `index_info()`.

```python
@dataclass
class IndexInfo:
    vector_field: str
    index_type: str              # e.g. "IVF_PQ"
    metric: str                  # "cosine" | "l2" | "dot"
    params: dict[str, Any]       # {"num_partitions": 256, "num_sub_vectors": 16, ...}
```

---

## Exceptions

All exceptions inherit from `ObjectVectorDBError`, which itself inherits from
`Exception`. `ObjectNotFound` additionally inherits from `KeyError` for
compatibility with dict-style callers.

| Exception                 | Raised by                                                                                      |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| `ObjectNotFound`          | `update`, `batch_update` when `object_id` does not exist.                                      |
| `DuplicateObject`         | `add`, `batch_add` when `object_id` already exists (or is duplicated in the batch).             |
| `VectorFieldNotRegistered`| `search`, `export_vectors`, `create_index`, `drop_index`, `rebuild_index` on unknown field.    |
| `DimensionMismatch`       | Any write or search where vector length ≠ registered dim.                                      |
| `SchemaError`             | Reserved-prefix property names, None-typed new columns, rename/drop conflicts, etc.            |
| `MetricMismatch`          | `search` when the caller passes a metric that disagrees with the existing index's metric.      |
