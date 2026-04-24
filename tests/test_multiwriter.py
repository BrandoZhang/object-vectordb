"""Concurrency tests for multi-writer correctness.

Each test uses threads to exercise race conditions that the single-writer
design could not survive. Assertions are on the final committed state, not
on intermediate call ordering (which is non-deterministic).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from object_vectordb import DuplicateObject, ObjectNotFound, ObjectVectorDB

# ---------------------------------------------------------------------------
# Concurrent add
# ---------------------------------------------------------------------------


def test_add_after_add_same_id_raises(tmp_path):
    # Sequential case: a second add() of the same id always raises DuplicateObject,
    # because the row is visible at merge_insert read time and no rows are inserted.
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")

    col.add("x", properties={"n": 1})
    with pytest.raises(DuplicateObject):
        col.add("x", properties={"n": 2})

    assert len(col.list_objects()) == 1


def test_concurrent_add_same_id_is_best_effort(tmp_path):
    """Document limitation: concurrent add() of the same id may produce duplicates.

    Lance treats a no-match merge_insert as a commutative append — two writers
    that both observe "no existing row at read time" will both commit, leaving
    two rows with the same object_id.  There is no primary-key constraint at
    commit time, and `MergeResult.num_inserted_rows` is populated from the
    writer's own snapshot.  Callers that need strict same-id uniqueness under
    concurrency must serialize externally (a lock per id, or a distributed
    lock across processes).
    """
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")

    results = []
    lock = threading.Lock()

    def try_add():
        try:
            col.add("x", properties={"n": 1})
            with lock:
                results.append("ok")
        except DuplicateObject:
            with lock:
                results.append("dup")

    threads = [threading.Thread(target=try_add) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # At least one insert succeeded; no unexpected exception types.
    assert "ok" in results
    assert set(results) <= {"ok", "dup"}
    assert col.exists("x")


def test_concurrent_add_distinct_ids(tmp_path):
    # Two threads adding different ids — both should succeed.
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")

    errors = []

    def try_add(oid):
        try:
            col.add(oid, properties={"n": int(oid)})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=try_add, args=(str(i),)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(col.list_objects()) == 10


# ---------------------------------------------------------------------------
# Concurrent update
# ---------------------------------------------------------------------------


def test_update_raises_for_deleted_row(tmp_path):
    # Sequential simulation: delete then update must raise ObjectNotFound.
    # (The multi-writer version is non-deterministic; the sequential case
    # pins the invariant that update() never silently no-ops on missing rows.)
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")

    col.add("x", properties={"title": "a"})
    col.delete("x")

    with pytest.raises(ObjectNotFound):
        col.update("x", properties={"title": "B"})

    assert col.get("x") is None


def test_concurrent_update_distinct_rows(tmp_path):
    # N threads each updating a distinct row — no conflicts, all must succeed.
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")
    n = 20
    for i in range(n):
        col.add(str(i), properties={"v": i})

    errors = []

    def bump(oid):
        try:
            col.update(oid, properties={"v": -1})
        except Exception as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(bump, str(i)) for i in range(n)]
        for f in as_completed(futs):
            f.result()  # re-raise any exception

    assert not errors
    for i in range(n):
        assert col.get(str(i)).properties["v"] == -1


# ---------------------------------------------------------------------------
# Concurrent upsert
# ---------------------------------------------------------------------------


def test_upsert_inserts_when_missing(tmp_path):
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")

    col.upsert("x", properties={"n": 42})
    obj = col.get("x")
    assert obj is not None
    assert obj.properties["n"] == 42


def test_concurrent_upsert_on_existing_row(tmp_path):
    # Multiple threads upserting the SAME pre-existing id — each upsert hits
    # `when_matched_update_all`, which conflict-retries via the Lance manifest.
    # Exactly one row must remain, with one of the writers' values.
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")
    col.add("x", properties={"n": -1})  # seed so all upserts are updates

    errors = []

    def do_upsert(n):
        try:
            col.upsert("x", properties={"n": n})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=do_upsert, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert col.exists("x")
    assert len(col.list_objects()) == 1
    assert col.get("x").properties["n"] in range(5)


# ---------------------------------------------------------------------------
# Concurrent schema mutation (register_vector_field)
# ---------------------------------------------------------------------------


def test_concurrent_register_distinct_fields(tmp_path):
    # Two threads each register a different vector field on the same collection.
    # Both fields must be visible after both threads complete.
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")

    errors = []

    def register(name, dim):
        try:
            col.register_vector_field(name, dim)
        except Exception as exc:
            errors.append((name, exc))

    t1 = threading.Thread(target=register, args=("alpha", 4))
    t2 = threading.Thread(target=register, args=("beta", 8))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors
    names = {f.name for f in col.list_vector_fields()}
    assert "alpha" in names
    assert "beta" in names


def test_concurrent_register_same_field_is_idempotent(tmp_path):
    # Two threads registering the same field with the same dim — both must succeed.
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")

    errors = []

    def register():
        try:
            col.register_vector_field("v", dim=4)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=register) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    fields = col.list_vector_fields()
    assert len(fields) == 1
    assert fields[0].dim == 4
