from __future__ import annotations


def _seed(store):
    store.register_vector_field("v", dim=2)
    for i in range(5):
        store.add(
            f"id{i}",
            properties={"n": i, "group": "even" if i % 2 == 0 else "odd"},
            vectors={"v": [float(i), 0.0]},
        )


def test_list_all(store):
    _seed(store)
    out = store.list()
    assert len(out) == 5
    assert {o.object_id for o in out} == {f"id{i}" for i in range(5)}


def test_list_with_where(store):
    _seed(store)
    out = store.list(where="n >= 3")
    ids = {o.object_id for o in out}
    assert ids == {"id3", "id4"}


def test_list_with_select_trims_properties(store):
    _seed(store)
    out = store.list(where="n = 2", select=["n"])
    assert len(out) == 1
    assert out[0].properties == {"n": 2}


def test_list_with_limit(store):
    _seed(store)
    out = store.list(limit=2)
    assert len(out) == 2


def test_list_with_offset(store):
    _seed(store)
    out_all = store.list(limit=5)
    out_offset = store.list(limit=5, offset=2)
    # Skip first two rows (order is not guaranteed, but len must differ)
    assert len(out_offset) == len(out_all) - 2
