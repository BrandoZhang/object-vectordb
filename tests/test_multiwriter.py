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


def test_concurrent_add_same_id(tmp_path):
    # Two threads both try to add the same id.  Exactly one should succeed;
    # the other must raise DuplicateObject.  Final row count must be 1.
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

    assert sorted(results) == ["dup", "ok"]
    assert col.exists("x")
    # Exactly one row in the table.
    assert len(col.list_objects()) == 1


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


def test_concurrent_upsert_same_id(tmp_path):
    # Multiple threads upserting the same id — exactly one row must exist
    # after all threads finish, with no exceptions.
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")

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
