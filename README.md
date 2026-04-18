# object-vectordb

[![CI](https://github.com/BrandoZhang/object-vectordb/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BrandoZhang/object-vectordb/actions/workflows/ci.yml)

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
from object_vectordb import ObjectVectorDB, ObjectUpdate, rrf_merge

db = ObjectVectorDB(uri="data/my_store")
media = db.collection("media")                 # opens or creates
notes = db.collection("notes")                 # sibling collection — isolated schema

# Register vector fields on a collection (zero-copy — existing rows keep their data,
# new column is null). Two collections at the same URI can register the SAME field
# name at DIFFERENT dims; they don't collide.
media.register_vector_field("text_openai", dim=1536, description="text-embedding-3-small")
media.register_vector_field("image_clip", dim=512)

# Add an object
media.add(
    "video_001",
    properties={"title": "A cat playing piano", "tags": ["cat", "piano"]},
    vectors={"text_openai": [...], "image_clip": [...]},
)

# Update — merge semantics; unspecified fields are untouched
media.update("video_001", properties={"views": 42000})
media.update("video_001", vectors={"text_openai": [...]})   # re-embed

# Clear a field on a specific object
media.update("video_001", properties={"description": None})
media.update("video_001", vectors={"image_clip": None})

# Search
hits = media.search(
    query_vector=[...],
    vector_field="text_openai",
    limit=10,
    metric="cosine",              # default: "cosine"
    where="tags LIKE '%cat%'",    # DataFusion SQL filter syntax
    select=["title"],
)
for h in hits:
    print(h.object_id, h.score, h.properties["title"])

# Multi-route retrieval via RRF (pure-Python utility)
r_text  = media.search(q_text,  vector_field="text_openai", limit=20)
r_image = media.search(q_image, vector_field="image_clip", limit=20)
merged  = rrf_merge(r_text, r_image, k=60, limit=10)

# Index management
media.create_index(
    "text_openai", index_type="IVF_PQ", metric="cosine",
    num_partitions=256, num_sub_vectors=16,
)
media.rebuild_index("text_openai")
media.drop_index("text_openai")

# Bulk export (skips rows with null vectors)
ids, embeddings = media.export_vectors("text_openai", where="views > 1000")
# embeddings: np.ndarray of shape (len(ids), 1536), float32

# DB-level collection management
db.list_collections()            # ["media", "notes"]
db.has_collection("media")       # True
db.drop_collection("notes")      # deletes the Lance table + registry entry
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
ObjectVectorDB             ── DB handle: open URI, create/list/drop collections
    │
    └── Collection         ── per-collection API (add, get, search, …)
            │
            ├── SchemaRegistry   ── JSON sidecar, namespaced by collection
            └── LanceDBBackend   ── all lancedb / pyarrow code
```

Collections are the unit of schema isolation: each collection owns its own vector
fields, property schema, and LanceDB table. Two collections at the same URI can
register the same field name with different dimensionalities without colliding.

Swapping backends (e.g. to Qdrant) means writing a new class with the same
`LanceDBBackend` method signatures and changing one import. There is no formal
plugin system — just disciplined layering.

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

## Docs

Deeper documentation lives in [`docs/`](docs/README.md):

- [Concepts](docs/concepts.md) — the mental model (objects, properties, vectors, registration).
- [Architecture](docs/architecture.md) — module layout, LanceDB usage, score
  conversion, null-clearing, concurrency.
- [API reference](docs/api.md) — full signature and behavior for every public
  method, dataclass, and exception.
- [Filter syntax](docs/filters.md) — DataFusion SQL `where=` examples and
  escaping guidance.
- [Testing](docs/testing.md) — what the 84-case suite covers and which tests
  gate v1.
