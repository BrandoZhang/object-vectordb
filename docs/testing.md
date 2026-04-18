# Testing

## Running

```bash
uv sync --all-extras
uv run pytest -q            # full suite
uv run pytest -q tests/test_search.py::test_search_dot_score_recovers_raw_dot
uv run ruff check src tests
```

The suite takes ~5 minutes end-to-end, dominated by LanceDB IVF_PQ index
creation tests. Individual files run in < 5 s.

## Layout

```
tests/
├── conftest.py               # tmp_path-based `store` fixture
├── test_lifecycle.py         # add / get / exists / delete / duplicate detection
├── test_update.py            # merge semantics + null-clear round-trip (the v1 gates)
├── test_batch.py             # add_many, batch_update across mixed column sets
├── test_vector_fields.py     # register, dim enforcement, drop, rename, reserved-prefix
├── test_schema.py            # schema() shape, auto-add per Python type
├── test_search.py            # per-metric calibration, where, select, metric mismatch
├── test_rrf.py               # rrf_merge determinism, k parameter, limit, tie-breaking
├── test_index.py             # create / rebuild / drop / index_info round-trip
├── test_export.py            # export_vectors aligned shapes, null skipping, where filter
├── test_list.py              # list() pagination + filters
└── test_persistence.py       # close + reopen preserves data, schema, and indices
```

## Gate tests (must pass before v1 is "done")

These tests cover the specific LanceDB gotchas encoded in the backend. A
regression in any of them means the underlying LanceDB behavior we rely on
has changed:

- `test_update.py::test_clear_property_to_none` — scalar null-clear via
  `values_sql` (mitigation for LanceDB #1325, #3105).
- `test_update.py::test_clear_vector_to_none` — vector null-clear via
  `merge_insert` with a FixedSizeList null batch.
- `test_search.py::test_search_dot_score_recovers_raw_dot` — calibration
  that `dot` metric returns `score = raw_dot_product`.
- `test_search.py::test_metric_mismatch_with_existing_index_raises` —
  `MetricMismatch` behavior when an index exists.
- `test_vector_fields.py::test_register_is_zero_copy` — adding a vector
  field preserves existing rows untouched; new column is null.
- `test_persistence.py::test_index_survives_reopen` — registry + index
  state survive process restart.
- `test_batch.py::test_batch_update_mixes_properties_and_vectors` — the
  column-signature grouping in `batch_update` actually preserves
  sibling-untouched columns.

## Fixtures

```python
@pytest.fixture
def store(tmp_path):
    return ObjectVectorDB(uri=str(tmp_path / "db"), table_name="objects")
```

Every test gets its own temp directory, so stores are fully isolated. No
shared state, no test ordering dependencies.

## Performance baselines (targets, not gates)

From the spec:

| Operation                                     | Target |
| --------------------------------------------- | ------ |
| `add` + `get` round-trip, single object       | < 10 ms |
| `search` with index on 100 K objects          | < 50 ms |
| `batch_update` 1 000 objects                  | < 5 s   |
| `export_vectors` 100 K objects                | < 10 s  |
| `register_vector_field` on 1 M-row table      | < 1 s  (zero-copy) |

These are not enforced by CI. Record observed numbers in the README when
relevant.

## Adding a test

1. Pick the file that matches the surface area you're testing (e.g.
   `test_update.py` for update-path bugs).
2. Use the existing `store` fixture — don't manually construct `tmp_path`.
3. Prefer testing the public API; only reach into `store._backend` or
   `store._registry` when exercising an invariant that cannot be observed
   through the public surface.
