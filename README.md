# object-vectordb

[![CI](https://github.com/BrandoZhang/object-vectordb/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BrandoZhang/object-vectordb/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BrandoZhang/object-vectordb/graph/badge.svg?branch=main)](https://codecov.io/gh/BrandoZhang/object-vectordb)

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
db.drop_collection("notes")      # deletes the Lance table
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
            ├── SchemaRegistry   ── Arrow field metadata on the Lance manifest
            └── LanceDBBackend   ── all lancedb / pyarrow code
```

Collections are the unit of schema isolation: each collection owns its own vector
fields, property schema, and LanceDB table. Two collections at the same URI can
register the same field name with different dimensionalities without colliding.

Swapping backends (e.g. to Qdrant) means writing a new class with the same
`LanceDBBackend` method signatures and changing one import. There is no formal
plugin system — just disciplined layering.

For the backend rationale (why LanceDB over Weaviate, Qdrant, Milvus, Chroma,
pgvector, Pinecone, Vespa) and how this library compares to LangChain
VectorStores / LlamaIndex / Haystack / FiftyOne, see
[docs/comparison.md](docs/comparison.md).

## Performance

Observed quick-tier benchmarks (`uv run pytest benches/ --benchmark-only -m "not full"`).
Hardware: x86-64, 16 cores @ 2.8 GHz, 22 GB RAM, local ext4 storage.
Vector dim = 1024 throughout (realistic embedding size; BGE-M3-class).
Post-BTREE-index: the backend auto-creates a `BTREE` scalar index on
`object_id` at table bootstrap, so existence-check paths are O(log N).

| Operation                                       | Spec target | Observed median | Status |
| ----------------------------------------------- | ----------- | ---------------: | ------ |
| `search`, indexed (IVF_PQ), 1 K rows             | < 50 ms @ 100 K   | 11.96 ms  | on track |
| `search`, brute-force, 1 K rows                  | < 50 ms @ 100 K   | 14.29 ms  | on track |
| `register_vector_field` on 10 K-row table        | < 1 s @ 1 M       | 29.62 ms  | on track (zero-copy) |
| `add` single object                              | < 10 ms (as part of roundtrip) | 29.58 ms | **over target** |
| `add` + `get` roundtrip, single object           | < 10 ms           | 54.54 ms  | **over target** |
| `get` single object (1 K-row table)              | — (part of roundtrip) | 361.16 ms | see note |
| `export_vectors` from 1 K-row table              | < 10 s @ 100 K    | 352.22 ms | on track |
| `batch_add` 1 000 rows                            | — (informational) | Lance OOM | see note |
| `batch_update` 1 000 rows (full tier; not re-run) | < 5 s            | not measured | — |

**Notes**:

- **`get` is still the slow outlier.** The BTREE cuts the id lookup to
  O(log N), but `get()` reconstructs a full object row including every
  registered vector column; at dim=1024 that's 4 KB/row plus the
  pyarrow→Python list conversion, which dominates the ~360 ms median.
  Returning a narrower projection (or deferring vector materialization)
  would close most of this gap — filed as a follow-up.
- **`add_single` and `add_get_roundtrip` improved substantially vs the
  previous dim=128 pre-BTREE run** (40 ms → 30 ms and 51 ms → 55 ms
  respectively) even with 8× larger vectors, confirming that the
  existence-check was the previous bottleneck, not vector I/O.
- **`batch_add` 1 000 rows hit Lance's DataFusion sort memory pool** at
  dim=1024:
  ```
  Not enough memory to continue external sort … 6.4 MB remain available
  for the total pool … merge_insert.rs
  ```
  A 1 000-row × 1024-dim batch is 4 MB; `merge_insert`'s external sort
  needs more headroom than Lance allocates by default. Workarounds: feed
  `batch_add` in chunks of ≤250 rows, or configure a larger memory pool
  via `storage_options` at connect time. The constraint scales with
  dim × batch_size, so this is a real production consideration for
  high-dim embeddings.
- **Spec-scale (100 K / 1 M row) numbers are still not recorded.** Run
  `uv run pytest benches/ --benchmark-only -m full` on a sustained-CPU
  machine with enough memory pool to collect them; expect `batch_update`
  to be the dominant cost.

Re-run and update this table whenever the benchmark suite is executed. See
[`benches/README.md`](benches/README.md) for the full recipe. The default
bench dim is set in `benches/conftest.py::DEFAULT_DIM`.

## Concurrency

Concurrent readers are always safe. Writes against distinct ids and
updates/upserts on existing rows are multi-writer safe; schema mutations
retry on manifest conflicts. Same-id concurrent inserts can produce
duplicate rows (Lance has no primary-key enforcement) — for strict
uniqueness under concurrency, put a serialization point in front of the
SDK. See [`docs/concurrency.md`](docs/concurrency.md).

## Development

```bash
uv sync --all-extras
uv run pytest -q          # full test suite
uv run ruff check src tests
```

## Releases

[SemVer 2.0](https://semver.org/spec/v2.0.0.html) for the human-facing
version (`MAJOR.MINOR.PATCH` in `pyproject.toml`); pre-release artifacts
from `main` use PEP 440 `0.X.Y.devN` (Python's accepted form).

- **Every PR** that touches `src/` or `pyproject.toml` must bump the version
  (CI's `version-check.yml` enforces strict-greater than `main`).
- **Every push to `main`** auto-builds an sdist + wheel as a GitHub
  pre-release (`v<base>.dev<run_number>`).
- **Formal releases** are cut by tagging `vX.Y.Z`; CI verifies the tag
  matches `pyproject.toml` and creates a GitHub Release.

See [`CHANGELOG.md`](CHANGELOG.md) for the per-version log.

## License

[Apache License 2.0](LICENSE). Copyright 2026 The object-vectordb Authors.
The `NOTICE` file at the repo root carries the attribution required by §4(d)
of the license.

## Docs

Deeper documentation lives in [`docs/`](docs/README.md):

- [Concepts](docs/concepts.md) — the mental model (objects, properties, vectors, registration).
- [Comparison](docs/comparison.md) — why LanceDB, how this library compares
  to other vector DBs and abstraction layers.
- [Architecture](docs/architecture.md) — module layout, LanceDB usage, score
  conversion, null-clearing, concurrency.
- [API reference](docs/api.md) — full signature and behavior for every public
  method, dataclass, and exception.
- [Filter syntax](docs/filters.md) — DataFusion SQL `where=` examples and
  escaping guidance.
- [Testing](docs/testing.md) — what the 84-case suite covers and which tests
  gate v1.
