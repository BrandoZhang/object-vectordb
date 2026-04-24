# Vector DB Landscape and Backend Justification

*Surveyed 2026-04. Re-audit this doc before major release or when backend-level decisions are revisited — the landscape moves fast.*

## 1. Purpose

This document answers three questions that the rest of the docs deliberately
sidestep:

1. Why is `object_vectordb` built on LanceDB and not Weaviate, Qdrant, Milvus,
   Chroma, pgvector, FAISS, Pinecone, or Vespa?
2. If a user already has LanceDB, what does this library add that direct
   LanceDB use does not?
3. How does this library relate to existing abstraction layers — LangChain
   `VectorStores`, LlamaIndex, Haystack, FiftyOne?

The conclusion is open. If a better backend or abstraction exists, we would
rather know than pretend otherwise. Section 6 lists LanceDB's weaknesses
honestly, and section 8 names the signals that would push us to switch.

## 2. Requirements

Six **hard** requirements, pulled from the library's actual use case (see
[`concepts.md`](concepts.md) and [`CLAUDE.md`](../CLAUDE.md)):

1. **Multiple named vectors per object.** A single object (say a video)
   carries a `text_openai`, `image_clip`, and `audio_clap` embedding side by
   side — not three parallel collections joined by an ID.
2. **Multimodality.** Text, image, audio, and video embeddings are
   first-class siblings on one object. The storage shape must not privilege a
   dominant modality.
3. **Vector-fusion search.** Given several named vectors, a caller must be
   able to fuse their rankings (RRF or equivalent) in a single query path
   without hand-rolling the merge. We ship this via
   [`fusion.py::rrf_merge`](../src/object_vectordb/fusion.py).
4. **Dynamic property schema.** Property columns appear on first write; no
   prior migration, no pre-declared schema.
5. **Zero-copy schema evolution.** Adding or dropping a column is
   metadata-only — existing rows are not rewritten.
6. **Permissive OSS license.** Apache-2.0 or MIT. No AGPL, no proprietary
   lock-in.

**Nice-to-haves** (tie-breakers, not disqualifiers):

- Embedded Python deployment (no server process).
- S3 / object-store native format.
- BM25 + vector hybrid (already covered by requirement 3 if we are permissive
  about what "fusion" means).
- Managed / SaaS option for ops-averse users.

Multimodality and vector-fusion are the differentiators most competing DBs
don't handle natively — this is where the named-vectors-per-object model
earns its keep.

## 3. Landscape

- **LanceDB** — Apache-2.0. Lance columnar format. Embedded Python. S3/GCS/
  local via one URI. Multi-vector per row via `FixedSizeList<float32, dim>`
  columns. DataFusion SQL filter engine. Rust core.
  ([lancedb.com](https://lancedb.com))
- **Weaviate** — BSD-3. Server (Go). Object-centric. GraphQL + REST.
  "Named vectors" per object added in v1.24 (2024). Ships hybrid BM25+vector
  with RRF server-side. ([weaviate.io](https://weaviate.io))
- **Qdrant** — Apache-2.0. Server (Rust). Named vectors per point. Strong
  payload-filter engine. Embedded mode is in-process but not persistent in
  the Python client; production path is the server.
  ([qdrant.tech](https://qdrant.tech))
- **Milvus** — Apache-2.0. Distributed server. Multi-vector per entity
  (up to 10, since v2.4) with server-side RRF and weighted fusion.
  Heavyweight ops (etcd, object store, query/index/data nodes).
  ([milvus.io](https://milvus.io))
- **Chroma** — Apache-2.0. Embedded Python, simple API. One embedding per
  record; multimodality requires side-by-side collections.
  ([trychroma.com](https://www.trychroma.com))
- **pgvector** — PostgreSQL License. Postgres extension. IVFFLAT + HNSW. One
  vector column per row is idiomatic; multi-vector requires schema design
  plus joins. Maxes out practically at ~10M rows per table before recall and
  latency degrade. ([github.com/pgvector/pgvector](https://github.com/pgvector/pgvector))
- **FAISS** — MIT. Bare vector-search library; no metadata, no primary keys.
  Incompatible with any property-bag requirement on its own.
  ([github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss))
- **Pinecone** — Proprietary SaaS. Managed. Single vector per record
  (namespaces partition, they don't add a second modality on the same ID).
  ([pinecone.io](https://www.pinecone.io))
- **Vespa** — Apache-2.0. JVM. Production search engine with first-class
  tensor/vector fields and rank profiles. Heavy to operate.
  ([vespa.ai](https://vespa.ai))
- **VikingDB** — Proprietary SaaS (ByteDance / Volcano Engine, also
  surfaced via BytePlus). Managed vector service. Based on the public
  Python SDK and docs, the data model is **single vector per record**
  with scalar-field filtering and text + vector hybrid search;
  multi-named-vectors-per-record and server-side cross-vector RRF are not
  documented. Storage is hosted on ByteDance TOS rather than arbitrary S3.
  If newer versions have added multi-vector support, public docs don't
  show it yet.
  ([github.com/volcengine/vikingdb-python-sdk](https://github.com/volcengine/vikingdb-python-sdk),
  [docs.byteplus.com/en/docs/VikingDB](https://docs.byteplus.com/en/docs/VikingDB/Overview))

**Honorable mentions.** *Turbopuffer* (proprietary; object-store-native,
serverless — architectural reference to watch), *Typesense* (GPL-3.0 +
commercial; text-first), *Marqo* (Apache-2.0 on OpenSearch), *Redis Vector*
(RSALv2, not Apache-2.0), *Deep Lake* (MPL-2.0, dataset-centric).

## 4. Capability matrix

### Hard requirements

| Requirement | LanceDB | Weaviate | Qdrant | Milvus | Chroma | pgvector | FAISS | Pinecone | Vespa | VikingDB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multiple named vectors per object | ✓ (N FSL columns) | ✓ (v1.24+) | ✓ (named vectors) | ✓ (v2.4+, up to 10) | ✗ (one per record) | partial (schema-by-hand) | ✗ (no metadata) | ✗ (one vector per record) | ✓ (tensor fields) | ✗ (one vector per record per public docs) |
| Multimodality on one object | ✓ (siblings on row) | ✓ | ✓ | ✓ | ✗ | partial (join) | ✗ | ✗ | ✓ | ✗ (parallel collections only) |
| Vector-fusion search (RRF et al.) | partial (client-side `rrf_merge`) | ✓ (hybrid+RRF) | partial (client) | ✓ (server-side RRF + weighted) | ✗ | ✗ | ✗ | partial (SaaS features) | ✓ (rank profiles) | partial (text + vector hybrid only; no multi-vector fusion documented) |
| Dynamic property schema | ✓ (auto-add) | ✓ | ✓ | partial (schema-declared) | partial (metadata dict) | ✗ (DDL) | ✗ | ✓ (metadata dict) | ✗ (schema-declared) | ✗ (schema-declared) |
| Zero-copy schema evolution | ✓ (Lance metadata) | partial | partial | partial | partial | ✗ (ALTER TABLE rewrite risk) | ✗ | partial | partial | unknown (not documented publicly) |
| Permissive OSS (Apache-2.0 / MIT / BSD) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ (proprietary SaaS) |

### Nice-to-haves

| Nice-to-have | LanceDB | Weaviate | Qdrant | Milvus | Chroma | pgvector | FAISS | Pinecone | Vespa | VikingDB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Embedded Python (no server) | ✓ | ✗ | partial (in-memory) | ✗ | ✓ | partial (w/ Postgres) | ✓ | ✗ | ✗ | ✗ |
| S3 / object-store native | ✓ | ✗ | ✗ | partial (MinIO) | ✗ | ✗ | ✗ | n/a (SaaS) | ✗ | partial (TOS, ByteDance-hosted) |
| BM25 + vector hybrid | partial (FTS experimental) | ✓ | partial | ✓ | ✗ | partial (tsvector) | ✗ | partial | ✓ | ✓ |
| Managed / SaaS | ✓ (LanceDB Cloud) | ✓ | ✓ | ✓ (Zilliz) | ✓ | ✓ (RDS et al.) | ✗ | ✓ | ✓ | ✓ |

Reading the tables: a ✗ on a hard-requirement row is disqualifying. A ✗ on a
nice-to-have row is not.

## 5. Why LanceDB specifically

- **Only Apache-2.0 backend that clears every hard requirement
  simultaneously.** Weaviate, Qdrant, and Milvus all satisfy multi-named-
  vectors + multimodality + fusion, but impose a server deployment or (in
  Milvus's case) a distributed one. They remain solid choices — they simply
  lose on the nice-to-haves that keep this library's operational surface at
  zero.
- **Multi-vector per row is native and cheap.** N `FixedSizeList<float32,
  dim_i>` columns live on a single Lance table. Adding one via
  `table.add_columns(pa.field(...))` is metadata-only, no row rewrite. This
  directly enables the multimodality requirement without a
  parallel-collection hack.
- **Fusion-friendly storage.** Because the named vectors all share one
  `object_id`, `rrf_merge` can fuse ranked lists without a cross-collection
  join. See [`fusion.py`](../src/object_vectordb/fusion.py).
- **Zero-copy schema evolution.** Lance's columnar format records
  `add_columns` / `drop_columns` as metadata. Adding a vector field on a
  10K-row table completes in ~10 ms (see
  [`README.md`](../README.md#performance)).
- **DataFusion SQL filters.** Arbitrary predicates at query time —
  `where="views > 1000 AND tags LIKE '%cat%'"` — not a restricted DSL.
- **Disk-based IVF / HNSW.** Indexes are memory-mapped, not RAM-resident.
  Modest hardware at 1M+ rows is viable.
- **MVCC update path.** Frequent writes don't force full index rebuilds
  (contrast: FAISS).
- **Nice-to-haves thrown in for free.** Embedded Python, S3-native, and
  local-file deployment are all available without extra work.

## 6. LanceDB weaknesses (honest)

These are first-class content, not footnotes. Each has a concrete workaround
or mitigation in this library, but a reader considering LanceDB without our
layer should know them:

- **Null round-trip bugs** — LanceDB issues **#1325** and **#3105**.
  `table.update(values={col: None})` does not reliably round-trip. We issue
  `values_sql={col: "CAST(NULL AS <sql_type>)"}` for scalars and
  `merge_insert` with an Arrow-null `FixedSizeList` for vector clears. See
  [`CLAUDE.md`](../CLAUDE.md) line 58.
- **No primary-key uniqueness at the storage layer.** LanceDB has no PK
  concept: `merge_insert(...).when_not_matched_insert_all()` commits as a
  commutative append, so two concurrent writers inserting the same
  `object_id` can both succeed and leave duplicate rows. As of 0.2.0 most
  write paths are multi-writer safe (distinct ids, updates, upserts
  against existing rows, schema mutations), but strict same-id uniqueness
  under concurrency still needs a serialization point **in front of** the
  SDK — a single write-service process, a sharded writer fleet, or an
  external distributed lock. See section 7 for how other vector DBs solve
  this, and [`concurrency.md`](concurrency.md) for the reference
  architecture.
- **No scalar index on `object_id` by default** → O(N) `get`, O(N²)
  `batch_update`. See [`README.md`](../README.md) lines 158–165.
- **Index metric wins at search.** Once an index exists, LanceDB silently
  uses its own metric and ignores the caller's. We raise `MetricMismatch`
  instead. See [`CLAUDE.md`](../CLAUDE.md) line 62.
- **`rename_field` on an indexed column orphans the index** unless
  explicitly rebuilt. We handle this in the backend. See
  [`CLAUDE.md`](../CLAUDE.md) line 63.
- **`merge_insert` nulls out missing columns** by default. A batch that
  mixes `{"n": 1}` and `{"v": [...]}` rows would null-out `v` on the first
  row without column-signature grouping. See [`CLAUDE.md`](../CLAUDE.md)
  line 59.
- **Vector field registry lives in Arrow field metadata on the Lance
  manifest**, not a separate file. Conflict-retried by LanceDB, no
  cross-file atomicity hazard, no separate storage-abstraction code path
  for object-store URIs.

## 7. Multi-writer handling across backends

Every mainstream vector DB solves the same-key coordination problem
*below* the SDK layer — inside a storage-owned process, not in the client
library. This section summarizes how each candidate backend handles it,
to make explicit what `object_vectordb` is giving up by using LanceDB and
what a system designer therefore has to build on top.

| System      | Storage model                            | PK uniqueness                           | Where writes serialize                                        |
|-------------|------------------------------------------|-----------------------------------------|---------------------------------------------------------------|
| Weaviate    | LSM + WAL per shard                      | Enforced via inverted index on `_id`    | Shard primary (Raft); all writes go through the leader        |
| Qdrant      | Per-segment RocksDB                      | `upsert`-only API; last-write-wins      | Per-segment write lock inside the shard process               |
| Milvus      | Log-structured, distributed              | Optional (`AutoID=false` + PK field)    | Proxy → MsgStream (Pulsar/Kafka) → DataNode pipeline          |
| Chroma      | SQLite / DuckDB + HNSW                   | `upsert`-only API                       | Single-process by design (embedded); no cross-process story   |
| Pinecone    | Managed, opaque                          | `upsert`-only API; last-write-wins      | Internal transaction layer (not exposed)                      |
| pgvector    | Postgres heap + B-tree + MVCC            | Full SQL `PRIMARY KEY` constraints      | MVCC + page-level latches, standard Postgres guarantees       |
| Vespa       | Proton content cluster, per-bucket       | Document id enforced per bucket         | Content node owns a bucket; writes serialized per bucket      |
| **LanceDB** | **Append-only columnar + manifest OCC**  | **None**                                | **Appends commute; updates conflict-retry. Inserts do not.**  |

A few observations worth stating explicitly:

- **No mainstream vector DB relies on client-side coordination** for
  same-id uniqueness. All of them push serialization into a
  storage-owned process (or the storage engine itself).
- **Weaviate, Qdrant, and Vespa all use a single-writer-per-shard model.**
  This is the same pattern the reference architecture in
  [`concurrency.md`](concurrency.md) recommends for `object_vectordb` —
  they just implement it *inside* the DB process rather than in front of
  it.
- **Most backends sidestep the problem at the API level** by only
  exposing upsert semantics (last-write-wins). `add()`'s "error on
  duplicate" behavior is actually uncommon; it is a database constraint
  (Postgres PK, Vespa document id), not a client-side rule.
- **pgvector is the cheap escape hatch** for projects that need strict PK
  uniqueness and are willing to give up Lance's columnar-storage
  advantages.
- **LanceDB's position is deliberate, not an oversight.** Its
  append-commutes-with-append property is exactly what makes concurrent
  ingestion from Spark / Ray / Dask cheap. The tradeoff is the one
  documented above.

## 8. Is there a better option?

No OSS backend today satisfies all six hard requirements simultaneously —
most stumble either on *embedded + dynamic-schema-without-DDL* or on
*multi-named-vectors-per-object as a first-class shape*. Turbopuffer is the
architectural reference for object-store-native storage, but it is
proprietary and still primarily single-vector-per-record.

Concrete switch signals:

| Signal | Switch to | Why |
| --- | --- | --- |
| Need strict same-id uniqueness under concurrency without a write-service router | Qdrant / Weaviate / pgvector | PK uniqueness is enforced by the storage layer itself (cost: give up the embedded, object-store-native story) |
| Scale blows past 10M per collection | Milvus | Cluster-native sharding, server-side fusion |
| Polyglot clients required | Qdrant / Weaviate | Stable HTTP + gRPC |
| Want managed ops, cost not primary | Pinecone | Zero-op SaaS (cost: lose multi-named-vector model) |
| Postgres-only stack | pgvector (+ pgvectorscale) | No extra infra (cost: multimodality becomes manual) |
| Text-search primary, vectors secondary | Vespa / Typesense | Mature text-ranking primitives |
| S3-native storage is gating, SaaS acceptable | Turbopuffer | Storage architecture leader |

## 9. Why a library on top, not raw LanceDB

Eight pain points a direct-LanceDB caller hits that this library removes:

1. **Arrow types leak into user code.** A caller has to spell
   `pa.field("__vec_text_openai", pa.list_(pa.float32(), 1536))` to add a
   vector column.
2. **No separation between property and vector columns.** A `list[float]`
   property can be silently mistaken for a vector; the library enforces the
   `__vec_` prefix internally and exposes two distinct API arguments.
3. **Dimension mismatches fail at search time, not write time.**
   [`exceptions.py::DimensionMismatch`](../src/object_vectordb/exceptions.py)
   catches this at `add` / `update`.
4. **Null round-trip bugs require per-write SQL casts** (see section 6).
   The backend generates these transparently.
5. **`merge_insert` semantics surprise users.** Grouping batches by column
   signature is necessary for correctness but non-obvious. See
   [`CLAUDE.md`](../CLAUDE.md) line 59.
6. **Index metric conflicts are silently honored** by LanceDB. We raise
   [`MetricMismatch`](../src/object_vectordb/exceptions.py) instead.
7. **No primary-key check.** Raw LanceDB lets duplicate `object_id` values
   coexist. The library does a `count_rows("object_id = 'x'")` pre-check on
   `add` / `update`.
8. **No collection concept.** LanceDB has tables; the library adds
   collections — a collection is the unit of schema isolation (own vector
   fields, own property columns, own registry namespace). Two collections
   at the same URI with the same vector-field name at different dims do
   not collide.

### Before / after

Raw LanceDB, adding a vector column and one row (≈15 lines):

```python
import lancedb
import pyarrow as pa

conn = lancedb.connect("data/store")
tbl = conn.create_table(
    "media",
    schema=pa.schema([
        pa.field("object_id", pa.string()),
        pa.field("title", pa.string()),
    ]),
)
tbl.add_columns(
    pa.field("__vec_text_openai", pa.list_(pa.float32(), 1536))
)
tbl.add([{
    "object_id": "video_001",
    "title": "A cat playing piano",
    "__vec_text_openai": embedding,
}])
```

With `object_vectordb` (≈3 lines):

```python
media = ObjectVectorDB("data/store").collection("media")
media.register_vector_field("text_openai", dim=1536)
media.add("video_001", properties={"title": "A cat playing piano"},
          vectors={"text_openai": embedding})
```

## 10. Competing abstraction libraries

LangChain `VectorStores`, LlamaIndex, Haystack, and FiftyOne all wrap vector
DBs. None of them model **multi-named-vectors-per-object with an extensible
property bag, usable across modalities, fusable at query time**:

- **LangChain / LlamaIndex / Haystack** are document-centric: one embedding
  per record, augmented with metadata. Multimodal objects mean side-by-side
  stores joined on an external ID; RRF across those stores is
  caller-implemented.
- **FiftyOne** is object-centric but for image/video samples specifically,
  coupled to a media-pipeline model. Vectors are bolted on via `Brain` and
  fusion is not first-class.

This library is not a duplicate of those. It models what they don't: a
generic object identified by `object_id`, carrying any number of named
vectors across any modalities, with a dynamic property bag, and a fusion
utility that operates on the ranked results.

## 11. Sources

- [VectorDBBench](https://github.com/zilliztech/VectorDBBench) — QPS / recall
  benchmarks across engines.
- [ann-benchmarks](https://ann-benchmarks.com) — recall-vs-QPS curves for
  OSS vector indexes.
- [LanceDB documentation](https://lancedb.github.io/lancedb/) — format,
  schema evolution, filter syntax.
- [Lance format paper / repo](https://github.com/lancedb/lance) —
  zero-copy schema claims.
- [Weaviate — Named Vectors](https://weaviate.io/blog/named-vectors) —
  multi-vector-per-object model and hybrid search.
- [Qdrant — Named Vectors](https://qdrant.tech/documentation/concepts/vectors/)
  — per-point named vectors.
- [Milvus — Multi-vector hybrid search](https://milvus.io/docs/multi-vector-search.md)
  — server-side RRF and weighted fusion.
- [Chroma documentation](https://docs.trychroma.com) — embedded
  single-vector model.
- [pgvector](https://github.com/pgvector/pgvector) — Postgres extension
  reference.
- [FAISS](https://github.com/facebookresearch/faiss) — index-only library.
- [Pinecone — Records and namespaces](https://docs.pinecone.io) —
  single-vector record model.
- [Vespa — Tensor fields](https://docs.vespa.ai/en/tensor-user-guide.html)
  — tensor / rank-profile model.
- [VikingDB (Volcano Engine)](https://www.volcengine.com/product/vikingdb)
  — managed multi-vector service, hybrid search, proprietary SaaS.
- [Turbopuffer](https://turbopuffer.com) — object-store-native vector DB
  reference.
- [LangChain VectorStores](https://python.langchain.com/docs/integrations/vectorstores/),
  [LlamaIndex](https://docs.llamaindex.ai),
  [Haystack](https://haystack.deepset.ai),
  [FiftyOne Brain](https://docs.voxel51.com/brain.html) — abstraction-layer
  references.
