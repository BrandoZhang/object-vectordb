# Concurrency and multi-writer deployments

> **TL;DR** `object_vectordb` is a client-side SDK layered on LanceDB, which
> does not enforce primary-key uniqueness at the storage layer. Concurrent
> writers targeting **distinct** ids are safe; concurrent writers targeting
> the **same** id can produce duplicate rows. For strict same-id uniqueness
> under concurrency, put a serialization point **in front of** the SDK —
> typically a single write-service process per URI. This document explains
> why, shows a reference architecture, and compares to how other vector DBs
> handle the same problem.

## What the SDK guarantees (and what it does not)

| Operation         | Distinct ids | Same id, row already exists            | Same id, row does not yet exist     |
|-------------------|--------------|----------------------------------------|-------------------------------------|
| `add()`           | safe         | safe — one writer raises `DuplicateObject` | **unsafe** — may produce duplicate rows |
| `update()`        | safe         | safe — Lance `UpdateConfig` conflict-retries | N/A — raises `ObjectNotFound`       |
| `upsert()`        | safe         | safe — Lance `UpdateConfig` conflict-retries | **unsafe** — may produce duplicate rows |
| `delete()`        | safe         | safe — Lance `UpdateConfig` conflict-retries | N/A — no-op                         |
| Schema mutation   | safe         | —                                      | —                                   |

The two **unsafe** cases trace back to a single fact: LanceDB's
`merge_insert(...).when_not_matched_insert_all()` is implemented as an
**append**, and appends in Lance are commutative — two writers whose read
snapshots both miss the target id will both commit, leaving two rows with
the same `object_id`. `MergeResult.num_inserted_rows` is populated from each
writer's own snapshot, so the SDK cannot detect the race after the fact.

Concrete timeline:

```
time   writer A                     writer B                     table version
────────────────────────────────────────────────────────────────────────────
t=0                                                              V    (empty)
t=1    add("x") reads snap V
       → x not present
t=2                                 add("x") reads snap V        ← B missed A
                                    → x not present
t=3    commit → V+1, inserts x                                   V+1  (1 row)
t=4                                 commit → V+2                 V+2  (2 rows!)
                                    (Lance treats this as a
                                     commutative append, no
                                     conflict is raised)
```

## S3 and other object stores: Lance commit coordination is required

Before anything else: if your URI is `s3://…` (or any other object store)
and more than one process writes to it, **you must configure Lance's
commit coordination** via `storage_options`. Lance writes its manifest as
plain `PutObject` by default, and two concurrent writers can both PUT the
same manifest version — the later write silently clobbers the earlier
one, losing rows and schema changes with no error on either side. This
is distinct from the same-id insert race above: it affects *all* writes,
including distinct-id inserts and updates that would otherwise be safe.

`storage_options` is passed through from `ObjectVectorDB`:

```python
db = ObjectVectorDB(
    uri="s3://my-bucket/my-db",
    storage_options={
        # Exact keys depend on your LanceDB version and the store; see the
        # LanceDB object-store docs for the current names.  The two
        # supported mechanisms are:
        #   (a) S3 conditional PUT (If-None-Match), or
        #   (b) an external commit lock (historically DynamoDB).
    },
)
```

If every writer goes through a single write-service process (the
reference architecture below), the concurrent-manifest-commit window does
not occur in practice and `storage_options` can be left empty. You are
then responsible for ensuring exactly one writer process per URI at a
time.

## Design decision: where the constraint lives

Concurrency correctness has to be enforced by the component that **owns**
the data. `object_vectordb` does not own the data — LanceDB does. Because
Lance has no primary-key concept, the SDK can only offer *cooperative*
serialization, which is by definition best-effort:

- An in-process `threading.Lock` only serializes within one Python process.
- A filesystem advisory lock (`fcntl.flock`) only serializes within one
  host, and is unreliable over many network filesystems.
- A distributed lock (Redis, etcd, ZooKeeper) works across machines but
  requires every writer to opt in to the same lock — a single misbehaving
  client that opens the Lance URI directly bypasses it.

For those reasons, the SDK does **not** ship with built-in cross-process
locking. The decision is deliberate: the SDK is honest about its foundation.
If your deployment needs strict same-id uniqueness, wrap the SDK in a
system component that owns the serialization — typically a single
write-service process per URI.

> **Rule of thumb.** If you can name the component that is allowed to write
> to your URI, and it is exactly one process, you are safe. If the answer
> is "any client that imports the SDK," you are not.

## Reference architecture: the write-service pattern

The simplest robust design: a single process owns all writes to a URI, and
every writer in the fleet sends write RPCs through it. Reads go directly
against the URI (or against read replicas).

```
┌────────────────────────────────────────────────────────────┐
│                     Application layer                      │
│         (any number of processes, threads, hosts)          │
└───────────────┬────────────────────────────┬───────────────┘
                │ writes (RPC)               │ reads (direct)
                ▼                            ▼
 ┌────────────────────────────┐   ┌────────────────────────────┐
 │      Write Service         │   │    Read handles            │
 │     (single process        │   │   ObjectVectorDB(uri=…)    │
 │      per URI; leader +     │   │   opened read-only         │
 │      standby for HA)       │   │                            │
 │                            │   └──────────────┬─────────────┘
 │  ┌──────────────────────┐  │                  │
 │  │  Serialization       │  │                  │
 │  │  (queue or lock —    │  │                  │
 │  │   at most one write  │  │                  │
 │  │   per id in flight)  │  │                  │
 │  └──────────┬───────────┘  │                  │
 │             ▼              │                  │
 │  ┌──────────────────────┐  │                  │
 │  │   object_vectordb    │  │                  │
 │  │   (the only writer)  │  │                  │
 │  └──────────┬───────────┘  │                  │
 └─────────────┼──────────────┘                  │
               ▼                                 ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                       LanceDB URI                           │
 │           (local FS, shared FS, or object store)            │
 └─────────────────────────────────────────────────────────────┘
```

### Minimum viable implementation

For most projects the "write service" can start very small. A single-process
HTTP server that funnels writes through an `asyncio.Lock` (or a plain
`threading.Lock`) is enough to guarantee same-id serialization:

```python
# write_service.py  (sketch — not production code)
import asyncio
from fastapi import FastAPI
from object_vectordb import ObjectVectorDB

app = FastAPI()
db = ObjectVectorDB(uri="s3://my-bucket/my-db")
col = db.collection("items")
write_lock = asyncio.Lock()

@app.post("/items/{object_id}")
async def upsert_item(object_id: str, body: dict):
    async with write_lock:
        col.upsert(
            object_id,
            properties=body.get("properties"),
            vectors=body.get("vectors"),
        )
    return {"ok": True}
```

Readers do not go through this service — they open the URI directly:

```python
reader_db = ObjectVectorDB(uri="s3://my-bucket/my-db")
results = reader_db.collection("items").search(vec, "embed", limit=10)
```

### Scaling beyond one writer

When throughput outgrows one process, two standard options:

1. **Shard by id.** Run N write services; route writes for `id` to writer
   `hash(id) % N`. Each writer owns a disjoint key space, so same-id races
   cannot happen.
2. **Leader + standby for HA only.** Use a lease-based leader election
   (etcd/ZooKeeper/Consul). Only the leader writes; the standby takes over
   on failover. Throughput does not scale, but availability does.

Writer-writer sharding is strictly simpler than true multi-leader replication
and covers the vast majority of deployments.

## How other vector DBs handle multi-writer

Every mainstream vector DB — Weaviate, Qdrant, Milvus, Chroma, Pinecone,
pgvector, Vespa — solves the same-key coordination problem **below** the
SDK layer, either in a storage-owned write process (Weaviate / Qdrant /
Vespa shard primary; Milvus Proxy + MsgStream pipeline) or in the storage
engine itself (pgvector inherits Postgres's full SQL `PRIMARY KEY`
constraints). Client libraries never enforce PK uniqueness.

The reference architecture above ports that same single-writer-per-shard
pattern into an application-level process, because Lance does not ship
with one. See [`docs/comparison.md § 7`](comparison.md#7-multi-writer-handling-across-backends)
for the full comparison table and per-backend commentary.

## Choosing an approach

| Requirement                                       | Recommended approach                          |
|---------------------------------------------------|-----------------------------------------------|
| Single-writer / single-process ingest             | Use `object_vectordb` directly; no extra work |
| Multi-threaded ingest, one process                | Wrap writes in a `threading.Lock`             |
| Multi-process ingest, same host                   | Funnel writes through one process             |
| Multi-host ingest, strict same-id uniqueness      | Write-service pattern (figure above)          |
| Need full SQL transactions + PK constraints       | pgvector or similar                           |

The SDK is intentionally a thin layer. Correctness under concurrency is a
system-level responsibility, and we have tried to make that boundary
explicit rather than hide it behind a lock that only half-works.
