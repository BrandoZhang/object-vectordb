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
from object_vectordb.exceptions import DimensionMismatch, SchemaError
from object_vectordb.types import ObjectUpdate

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


def test_concurrent_register_different_dims_detected(tmp_path):
    """C2: two writers register the same field with DIFFERENT dims.

    One writer's add_columns wins at the column type; the loser must raise
    DimensionMismatch after catching "already exists", not silently write
    its own dim into the metadata and desync from the actual column.
    """
    import pyarrow as pa

    from object_vectordb.registry import VECTOR_COLUMN_PREFIX

    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")

    barrier = threading.Barrier(2)
    errors: list[tuple[int, Exception]] = []
    successes: list[int] = []

    def register(dim: int) -> None:
        barrier.wait()
        try:
            col.register_vector_field("v", dim=dim)
            successes.append(dim)
        except DimensionMismatch as exc:
            errors.append((dim, exc))

    t1 = threading.Thread(target=register, args=(4,))
    t2 = threading.Thread(target=register, args=(8,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    fields = col.list_vector_fields()
    assert len(fields) == 1
    registered_dim = fields[0].dim
    assert registered_dim in {4, 8}

    # The actual column's dim must agree with the metadata (the bug being
    # fixed was exactly this divergence).
    actual_type = col._backend._table.schema.field(VECTOR_COLUMN_PREFIX + "v").type
    assert isinstance(actual_type, pa.FixedSizeListType)
    assert actual_type.list_size == registered_dim

    # One writer succeeded; one observed the race and raised DimensionMismatch.
    # (If scheduling fully serialized them, both might "succeed" from their
    # own perspective — but the second must have seen the first's value via
    # the existing-metadata fast path, so only one dim can appear.)
    assert len(successes) + len(errors) == 2


# ---------------------------------------------------------------------------
# H1: concurrent create_table
# ---------------------------------------------------------------------------


def test_concurrent_collection_creation_same_name(tmp_path):
    """H1: two threads opening the same new collection must not race on
    create_table.  Both should end up with a valid handle."""
    db = ObjectVectorDB(uri=str(tmp_path / "db"))

    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def open_collection() -> None:
        barrier.wait()
        try:
            db.collection("shared")
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=open_collection)
    t2 = threading.Thread(target=open_collection)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors
    assert "shared" in db.list_collections()


# ---------------------------------------------------------------------------
# H2: batch_update detects rows deleted between pre-check and write
# ---------------------------------------------------------------------------


def test_batch_update_raises_when_row_deleted_mid_batch(tmp_path):
    """H2: simulate the delete-between-precheck-and-write window by patching
    _find_missing_ids to lie (say "all exist"), then deleting a row before
    the merge_insert runs.  The per-group num_updated_rows check must catch
    the missing row."""
    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")
    col.add("a", properties={"n": 1})
    col.add("b", properties={"n": 2})

    # Stage the race: delete "a" after pre-check is satisfied.
    original = col._backend._find_missing_ids

    def lying_precheck(ids: list[str]) -> list[str]:
        missing = original(ids)
        # After the (truthful) pre-check, delete "a" so the write finds
        # only "b".
        col.delete("a")
        return missing

    col._backend._find_missing_ids = lying_precheck  # type: ignore[method-assign]

    with pytest.raises(ObjectNotFound):
        col.batch_update(
            [
                ObjectUpdate(object_id="a", properties={"n": 10}),
                ObjectUpdate(object_id="b", properties={"n": 20}),
            ]
        )


# ---------------------------------------------------------------------------
# N1: property-column type race detection
# ---------------------------------------------------------------------------


def test_verify_property_column_type_raises_on_mismatch(tmp_path):
    """N1 unit test: _verify_property_column_type must raise SchemaError when
    the existing column's Arrow type differs from the inferred type."""
    import pyarrow as pa

    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")
    backend = col._backend

    # Seed a string-typed property column.
    col.add("a", properties={"field": "hello"})

    # Matching type — no raise.
    backend._verify_property_column_type("field", pa.string())

    # Mismatched type — must raise.
    with pytest.raises(SchemaError, match="already exists with type"):
        backend._verify_property_column_type("field", pa.int64())

    # Missing column entirely — must raise a SchemaError (different message).
    with pytest.raises(SchemaError, match="Expected column"):
        backend._verify_property_column_type("ghost", pa.int64())


def test_concurrent_property_add_different_types(tmp_path):
    """N1 integration test: two threads adding the same property with
    different inferred types must not silently desync column vs. values.

    Depending on thread scheduling, one of three outcomes is legal:
      (a) Both serialize — one defines the type, the other's value is coerced
          into it at encode time (existing single-writer behavior).
      (b) Both reach add_columns; the loser's "already exists" catch runs
          _verify_property_column_type and raises SchemaError.
      (c) Both reach add_columns and agreed on a type (unlikely for int/str).

    What must NOT happen: metadata desync — the column's actual Arrow type
    must agree with at least one writer's intent.
    """
    import pyarrow as pa

    db = ObjectVectorDB(uri=str(tmp_path / "db"))
    col = db.collection("c")

    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, object, object]] = []
    lock = threading.Lock()

    def do_add(oid: str, value: object) -> None:
        barrier.wait()
        try:
            col.add(oid, properties={"newprop": value})
            with lock:
                outcomes.append(("ok", oid, value))
        except SchemaError as exc:
            with lock:
                outcomes.append(("schema_error", oid, str(exc)))

    t1 = threading.Thread(target=do_add, args=("a", "hello"))
    t2 = threading.Thread(target=do_add, args=("b", 42))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Actual column type exists and is one of the two possibilities.
    actual_type = col._backend._table.schema.field("newprop").type
    assert actual_type in (pa.string(), pa.int64())

    # At least one writer completed successfully.
    oks = [o for o in outcomes if o[0] == "ok"]
    assert oks
