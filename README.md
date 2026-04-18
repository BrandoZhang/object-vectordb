# object-store

Multimodal object store built on [LanceDB](https://lancedb.com). Manages objects with
an evolving dynamic property schema and multiple named embedding vectors per object.

Design goals:

- **Object-centric** API — callers think in terms of objects, properties, and vectors;
  never tables, columns, or indexes.
- **Explicit properties vs. vectors** — two distinct namespaces, each with their own
  method arguments. No ambiguity about whether a `list[float]` property is a vector.
- **Registered vector fields** — declare a vector field's name and dimensionality before
  writing; dimension mismatches fail loudly at write time.
- **Zero-copy schema evolution** — add vector fields or drop columns without rewriting
  existing rows (LanceDB `add_columns` / `drop_columns` under the hood).
- **No embedding models, no RAG** — all vectors are provided by the caller.

## Install

```bash
uv sync --all-extras
```

## Quick start

```python
from object_store import ObjectStore, ObjectUpdate

store = ObjectStore(uri="data/my_store", table_name="media")

# Register vector fields (zero-copy — existing rows keep their data, new column is null)
store.register_vector_field("text_openai", dim=1536, description="text-embedding-3-small")
store.register_vector_field("image_clip", dim=512)

# Add an object
store.add(
    "video_001",
    properties={"title": "A cat playing piano", "tags": ["cat", "piano"]},
    vectors={"text_openai": [...], "image_clip": [...]},
)

# Update — merge semantics; unspecified fields are untouched
store.update("video_001", properties={"views": 42000})
store.update("video_001", vectors={"text_openai": [...]})   # re-embed

# Clear a field on a specific object
store.update("video_001", properties={"description": None})
store.update("video_001", vectors={"image_clip": None})

# Search
hits = store.search(
    query_vector=[...],
    vector_field="text_openai",
    limit=10,
    metric="cosine",              # default: "cosine"
    where="tags LIKE '%cat%'",    # DataFusion SQL filter syntax
    select=["title"],
)
for h in hits:
    print(h.object_id, h.score, h.properties["title"])

# Multi-route retrieval via RRF
r_text  = store.search(q_text,  vector_field="text_openai", limit=20)
r_image = store.search(q_image, vector_field="image_clip", limit=20)
merged  = ObjectStore.rrf_merge(r_text, r_image, k=60, limit=10)

# Index management
store.create_index(
    "text_openai", index_type="IVF_PQ", metric="cosine",
    num_partitions=256, num_sub_vectors=16,
)
store.rebuild_index("text_openai")
store.drop_index("text_openai")

# Bulk export (skips rows with null vectors)
ids, embeddings = store.export_vectors("text_openai", where="views > 1000")
# embeddings: np.ndarray of shape (len(ids), 1536), float32
```

## Filter syntax

The `where` parameter passes through to LanceDB's DataFusion SQL engine:

```python
where="views > 1000 AND tags LIKE '%cat%'"
where="status IN ('published', 'review')"
where="score BETWEEN 0.5 AND 0.9"
where="array_has(tags, 'cat')"
```

User-supplied input must be escaped by the caller; the library does not validate SQL.

## Score semantics

`SearchResult.score` is always a similarity (higher = more similar). It is derived from
LanceDB's `_distance` as follows:

| `metric` | Formula                  | Range       | Notes |
| -------- | ------------------------ | ----------- | ----- |
| `cosine` | `1 - distance`           | `[-1, 1]`   | Exact cosine similarity. |
| `l2`     | `1 / (1 + distance)`     | `(0, 1]`    | Monotonic only; do not compare magnitudes across different query vectors. |
| `dot`    | `1 - distance`           | unbounded   | Equals raw dot product (LanceDB stores `_distance = 1 - dot`). |

If a vector field has an index, `search()` enforces that the requested `metric` matches
the index's metric — otherwise it raises `MetricMismatch`. Drop and recreate the index
to change metrics.

## Architecture

```
ObjectStore                 ── public API, Python-native types only
    │
    ├── SchemaRegistry      ── JSON sidecar at <uri>/object_store_registry.json
    └── LanceDBBackend      ── all lancedb / pyarrow code
```

Swapping backends (e.g. to Qdrant) means writing a new backend class with the same
method signatures and replacing the import in `store.py`. There is no formal plugin
system — just a single concrete backend today and disciplined layering.

## Concurrency

Single-writer. Concurrent readers are safe. The JSON registry sidecar is not locked.
For multi-writer setups, route writes through a single worker process.

## Development

```bash
uv sync --all-extras
uv run pytest -q          # full test suite
uv run ruff check src tests
```

## License

MIT (or project default).
