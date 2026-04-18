# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`object_vectordb` is a Python library that layers an object-centric API over [LanceDB](https://lancedb.com). Callers work in terms of "objects" with a dynamic property bag and named embedding vectors; they never see LanceDB tables, columns, or pyarrow types.

## Commands

Dependencies, lint, format, tests, and build all go through `uv`:

```bash
uv sync --all-extras              # install runtime + dev deps into .venv
uv run pytest -q                   # full test suite (~5 min, LanceDB IVF index tests dominate)
uv run pytest tests/test_search.py # a single file
uv run pytest tests/test_search.py::test_search_dot_score_recovers_raw_dot -q   # a single test
uv run ruff check src tests
uv run ruff format src tests       # apply (CI runs --check)
uv build                           # sdist + wheel into dist/
```

CI (`.github/workflows/ci.yml`) runs lint, a pytest matrix over Python 3.10/3.11/3.12, and a build on every PR and push to `main`. Keep the `ruff format --check` gate green — run `uv run ruff format src tests` before committing.

## Architecture — the layering rule

```
ObjectVectorDB    (db.py)            ← DB handle: open URI, create/list/drop collections
    │
    └── Collection  (collection.py)  ← per-collection API: add, get, search, register_vector_field…
            │
            ├── SchemaRegistry       ← JSON sidecar, scoped per-collection via CollectionRegistry
            └── LanceDBBackend       ← all lancedb + pyarrow code; one backend per Collection
```

- **Collections** are the unit of schema isolation: each collection owns its own vector fields, property columns, and LanceDB table. Collections at the same URI never share state.
- `rrf_merge` is a module-level function in `fusion.py`, not a method on any class — it's a pure utility over `SearchResult` lists.

**Hard rule: `db.py` and `collection.py` must not import `pyarrow`, and only `db.py` may import `lancedb` (for the initial `connect`).** Every storage-engine specific call lives in `backend.py`. Swapping backends should be a single-file change.

## Architecture — big picture

**Properties vs. vectors are separate namespaces.** The API takes them as distinct arguments. Internally, vector columns are prefixed `__vec_<name>`; property names are rejected if they start with that prefix. Property names otherwise are stored as-is.

**Vector fields must be registered before use, per collection.** `collection.register_vector_field(name, dim)` adds a zero-copy `FixedSizeList<float32, dim>` column via LanceDB's `table.add_columns(pa.field(...))`. Registration is tracked in a JSON sidecar at `<uri>/object_vectordb_registry.json`, namespaced by collection name — NOT inside the Lance table. The registry is the source of truth for which collections exist, which columns are vectors vs. properties within each collection, each vector's dim, and each vector's index config.

**Schema grows automatically for properties.** Property columns are added on first write via Arrow-type inference from the sample value (`arrow_utils.python_value_to_arrow_type`). `None` alone cannot be inferred and raises `SchemaError`.

**Merge-update semantics.** `update()` and `batch_update()` only touch fields the caller passes. `None` clears a field on a specific object (the column stays, the cell becomes null).

## Non-obvious LanceDB behaviors we encode

These are load-bearing — changing them quietly will break tests and correctness:

- **Null round-trip bugs (LanceDB #1325, #3105).** `table.update(values={col: None})` does not reliably round-trip. We use `values_sql={col: "CAST(NULL AS <sql_type>)"}` for scalar null-clearing and `merge_insert` with an Arrow-null FixedSizeList for vector null-clearing. The `arrow_type_to_sql_type` mapping lives in `arrow_utils.py`.
- **`batch_update` groups rows by column signature.** A single `merge_insert(...).when_matched_update_all()` call treats missing source columns as null overwrites. Without grouping, a row that specifies only `{"n": 1}` in the same batch as a sibling that sets `{"v": [...]}` would null-out `v` on the first row. See `_apply_update` and the grouping in `batch_update` in `backend.py`.
- **No auto-column-infer on `table.add`.** The backend always calls `add_columns(pa.field(...))` before an insert that references a new property or vector column.
- **No primary-key uniqueness.** `add()` does a `count_rows("object_id = 'x'")` pre-check and raises `DuplicateObject`. `update()` does the same pre-check and raises `ObjectNotFound`. Race-prone under concurrent writers — documented as single-writer only.
- **Index metric wins at search time.** Once an index exists, LanceDB ignores the caller's `distance_type()`. We read the stored metric from the registry and raise `MetricMismatch` if the caller passes a conflicting `metric=`. Do not soften this to a warning.
- **Renaming an indexed vector column orphans the index.** `rename_field()` drops the index, renames the column, then recreates the index using the stored config.

## Score conversion

`SearchResult.score` is a similarity (higher = better) derived from LanceDB's `_distance`:

| metric   | conversion              | notes |
| -------- | ----------------------- | ----- |
| `cosine` | `1 - distance`          | exact cosine similarity |
| `l2`     | `1 / (1 + distance)`    | monotonic only; magnitudes not meaningful |
| `dot`    | `1 - distance`          | equals raw dot product (LanceDB stores `_distance = 1 - dot`, verified) |

Logic is in `scoring.py`. If you add a new metric, add a calibration test in `test_search.py` using vectors with known similarities.

## Where to read more

- `docs/concepts.md` — mental model (objects, properties, vectors, registration).
- `docs/architecture.md` — design, LanceDB API usage table, full gotcha list, write-path walkthroughs for add / update / batch_update.
- `docs/api.md` — full public API reference.
- `docs/filters.md` — DataFusion SQL `where=` syntax.
- `docs/testing.md` — what each test file covers and the v1 gate tests.

## Branch and PR conventions

- Develop on feature branches (current active branch: `claude/multimodal-object-store-ryKw0`). Never force-push to `main`.
- CI must be green before merging. The format check is part of lint.
- The commit trailer `https://claude.ai/code/session_...` is added automatically when Claude Code creates commits — leave it in place.
