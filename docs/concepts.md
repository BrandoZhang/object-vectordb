# Core Concepts

This page explains the mental model the API is built around. Reading it should
take five minutes and leave you with a clear understanding of what a
"collection," an "object," a "property," and a "vector" mean in this library.

## Database and collections

The top-level handle is `ObjectVectorDB(uri)`. It does not hold any objects
itself — it's a reference to a directory where one or more **collections**
live:

```python
from object_vectordb import ObjectVectorDB

db = ObjectVectorDB(uri="data/my_store")
videos = db.collection("videos")          # opens or creates
images = db.collection("images")          # sibling collection; isolated schema
```

A collection is a named group of objects with its own vector-field registry,
its own property schema, and its own underlying LanceDB table. Collections
at the same URI never share state: a vector field registered on `videos` is
invisible to `images`, and the same field name can exist at different
dimensionalities in each collection.

All per-object operations (`add`, `get`, `search`, `register_vector_field`,
`create_index`, …) live on the collection. DB-level operations are limited
to:

```python
db.list_collections()        # ["images", "videos"]
db.has_collection("videos")  # True
db.drop_collection("images") # deletes the Lance table
```

## Object

An Object is identified by a unique string `object_id` (e.g. a video asset ID).
Each Object carries two kinds of fields:

- **Properties** — any scalar or structured annotation: strings, numbers,
  booleans, lists, dicts, bytes. The library does not restrict property types;
  it infers Arrow types on first write and passes values through to the
  backend. Properties can be added, updated, or cleared at any time; the
  underlying column is created on demand.
- **Vectors** — pre-computed embedding arrays. Each vector field has a name
  (e.g. `text_openai`, `image_clip`, `audio_clap`) and a fixed
  dimensionality. Vectors are always provided by the caller — the library
  never calls any embedding model.

Properties and vectors are **explicitly separated** in the API. They occupy
distinct namespaces and are passed as separate arguments. This eliminates the
classic ambiguity: a property that happens to be a list of floats is never
mistaken for a vector.

```python
videos.add(
    "video_001",
    properties={"title": "A cat playing piano", "views": 42},
    vectors={"text_openai": [0.12, 0.08, ...],  # dim=1536
             "image_clip":  [0.51, 0.63, ...]}, # dim=512
)
```

## Why register vector fields?

Before writing vectors of a new type, the caller **registers** the vector
field on the collection, declaring its name and dimensionality:

```python
videos.register_vector_field("text_openai", dim=1536,
                             description="text-embedding-3-small on title")
```

This is a schema-level operation (zero-copy in LanceDB — no data rewrite).
Once registered, the field is available on all objects, defaulting to `None`
for existing rows.

Registration serves two purposes:

1. **Write-time validation.** If a caller passes a vector with the wrong
   dimensionality, the error is caught immediately at write time with a
   clear message (`DimensionMismatch: Vector 'text_openai' expects dim=1536,
   got dim=1024`), not silently stored and discovered later at search time.
2. **Auditability.** In production you want to know *when* a vector field
   was added and *by whom*. An explicit registration call provides a clear
   code point to trace.

For prototyping, pass `auto_register=True` when opening a collection
(`db.collection("name", auto_register=True)`); the first vector written
under a previously-unseen name is implicitly registered with
`dim=len(vector)`. Explicit registration is recommended for production use.

### Properties and vectors are independent

The library does **not** couple a vector to any specific property. The
relationship between properties and vectors is many-to-many in practice:

- One property may produce multiple vectors (the same text with different
  embedding models — `text_openai`, `text_bge`, `text_cohere`).
- One vector may derive from multiple properties (concatenated title +
  description + tags).
- Some vectors correspond to no property at all (visual frame embeddings
  extracted from a binary asset).

If you need to track provenance ("this vector came from the `description`
field via OpenAI `text-embedding-3-small`"), use the optional `description`
parameter on `register_vector_field()`, or maintain an external mapping.

## Index

Each registered vector field can have an independent ANN index with its own
parameters. Indexes are created and rebuilt explicitly via
[`create_index`](api.md#create_index), [`rebuild_index`](api.md#rebuild_index),
and [`drop_index`](api.md#drop_index).

The distance metric (cosine, L2, dot) is specified at index creation time and
at search time. If a field has an index and `search()` is called with a
different metric, the library raises `MetricMismatch` — LanceDB silently uses
the index's metric in that case, so we surface it as a hard error.

For neighborhood queries ("every object within distance D", deduplication,
"similar but not identical") use
[`search_within`](api.md#search_within) instead of top-k `search`. Pass
`exact=True` when completeness matters, since radius queries against an
IVF index are approximate by default.

## Merge-update semantics

`update()` only touches the fields you specify:

```python
# Update title, leave vectors and other properties untouched
store.update("video_001", properties={"title": "new"})

# Re-embed after title change, leave properties untouched
store.update("video_001", vectors={"text_openai": [...new embedding...]})
```

Passing `None` for a field **clears** that field on that specific object:

```python
# Clear a property
store.update("video_001", properties={"description": None})

# Clear a vector
store.update("video_001", vectors={"image_clip": None})
```

Clearing preserves the schema — the column still exists, the cell becomes
null. To remove a column entirely, use [`drop_fields`](api.md#drop_fields).

## Schema evolution

Adding new properties or vector fields is zero-copy: LanceDB records the
schema change as metadata and leaves existing data untouched. Existing rows
read back with `None` for the new field. Dropping a column is also zero-copy.

Renaming a column preserves data; if the column is an indexed vector field,
the index is dropped and recreated (LanceDB's indices reference column names,
so renaming without rebuilding would orphan the index).

```python
store.register_vector_field("audio_clap", dim=256)   # zero-copy
store.drop_fields(["obsolete_annotation"])           # zero-copy
store.rename_field("text_openai", "text_oai_v1")     # zero-copy (+ index rebuild if indexed)
```

## Filter syntax

The `where=` parameter on `search()`, `list_objects()`, and `export_vectors()` is a
**DataFusion SQL** expression. See [filters.md](filters.md) for the full
reference and examples.

## Concurrency

Concurrent readers are always safe. Writes against distinct ids, and
updates / upserts against the same *existing* row, are multi-writer safe;
schema mutations retry on manifest conflicts. The one remaining caveat is
same-id concurrent inserts, which can produce duplicate rows because
Lance has no primary-key enforcement. For strict same-id uniqueness under
concurrency, put a serialization point in front of the SDK — typically a
single write-service process per URI. See `docs/concurrency.md` for the
full design discussion and a reference architecture.

## What the library does NOT do

- No embedding model integration. All vectors are provided by the caller.
- No RAG pipeline. No LLM calls.
- No cross-object relationships. No foreign keys, no graph traversal.
- No REST API. Python library only.
- No backend-agnostic filter DSL. The `where=` parameter passes through to
  the backend's native filter engine.
