from __future__ import annotations

import pytest

from object_vectordb import DuplicateObject, ObjectAdd, ObjectNotFound, ObjectUpdate


def test_batch_add_inserts_all(store):
    store.register_vector_field("v", dim=2)
    store.batch_add(
        [
            ObjectAdd(object_id="a", properties={"n": 1}, vectors={"v": [1.0, 0.0]}),
            ObjectAdd(object_id="b", properties={"n": 2}, vectors={"v": [0.0, 1.0]}),
            ObjectAdd(object_id="c", properties={"n": 3}, vectors={"v": [1.0, 1.0]}),
        ]
    )
    assert store.get("a").properties["n"] == 1
    assert store.get("b").properties["n"] == 2
    assert store.get("c").properties["n"] == 3


def test_batch_add_rejects_duplicates_in_batch(store):
    with pytest.raises(DuplicateObject):
        store.batch_add(
            [
                ObjectAdd(object_id="a", properties={"n": 1}),
                ObjectAdd(object_id="a", properties={"n": 2}),
            ]
        )


def test_batch_add_rejects_existing_id(store):
    store.add("a", properties={"n": 1})
    with pytest.raises(DuplicateObject):
        store.batch_add([ObjectAdd(object_id="a", properties={"n": 99})])


def test_batch_update_mixes_properties_and_vectors(store):
    store.register_vector_field("v", dim=2)
    for oid in ["a", "b", "c"]:
        store.add(oid, properties={"n": 0}, vectors={"v": [0.0, 0.0]})

    store.batch_update(
        [
            ObjectUpdate(object_id="a", properties={"n": 1}),
            ObjectUpdate(object_id="b", vectors={"v": [1.0, 1.0]}),
            ObjectUpdate(object_id="c", properties={"n": 3}, vectors={"v": [2.0, 2.0]}),
        ]
    )
    assert store.get("a").properties["n"] == 1
    assert store.get("a").vectors["v"] == pytest.approx([0.0, 0.0])
    assert store.get("b").properties["n"] == 0
    assert store.get("b").vectors["v"] == pytest.approx([1.0, 1.0])
    assert store.get("c").properties["n"] == 3
    assert store.get("c").vectors["v"] == pytest.approx([2.0, 2.0])


def test_batch_update_null_clears(store):
    store.register_vector_field("v", dim=2)
    store.add("a", properties={"n": 1}, vectors={"v": [1.0, 2.0]})
    store.add("b", properties={"n": 2}, vectors={"v": [3.0, 4.0]})

    store.batch_update(
        [
            ObjectUpdate(object_id="a", properties={"n": None}),
            ObjectUpdate(object_id="b", vectors={"v": None}),
        ]
    )
    assert store.get("a").properties["n"] is None
    assert store.get("b").vectors["v"] is None


def test_batch_update_on_missing_id_raises(store):
    store.add("a", properties={"n": 1})
    with pytest.raises(ObjectNotFound):
        store.batch_update([ObjectUpdate(object_id="ghost", properties={"n": 2})])


def test_batch_update_rejects_duplicate_in_batch(store):
    store.add("a", properties={"n": 1})
    with pytest.raises(DuplicateObject):
        store.batch_update(
            [
                ObjectUpdate(object_id="a", properties={"n": 2}),
                ObjectUpdate(object_id="a", properties={"n": 3}),
            ]
        )
    assert store.get("a").properties["n"] == 1


def test_batch_update_large(store):
    store.register_vector_field("v", dim=3)
    for i in range(100):
        store.add(f"id{i}", properties={"n": i}, vectors={"v": [float(i), 0.0, 0.0]})
    store.batch_update(
        [ObjectUpdate(object_id=f"id{i}", properties={"n": i * 10}) for i in range(100)]
    )
    for i in range(100):
        assert store.get(f"id{i}").properties["n"] == i * 10
