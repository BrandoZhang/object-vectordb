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
uv run ruff check src tests benches
uv run ruff format src tests benches   # apply (CI runs --check)
uv build                           # sdist + wheel into dist/

uv run pytest benches/ --benchmark-only -m "not full"   # quick benchmark tier (~30 s)
uv run pytest benches/ --benchmark-only -m full         # spec-scale benchmarks (~5–10 min)
```

CI (`.github/workflows/ci.yml`) runs lint, a pytest matrix over Python 3.10/3.11/3.12, and a build on every PR and push to `main`. Keep the `ruff format --check` gate green — run `uv run ruff format src tests benches` before committing.

## Architecture — the layering rule

```
ObjectVectorDB    (db.py)            ← DB handle: open URI, create/list/drop collections
    │
    └── Collection  (collection.py)  ← per-collection API: add, get, search, register_vector_field…
            │
            ├── SchemaRegistry       ← Arrow field metadata on the Lance manifest, scoped per-collection via CollectionRegistry
            └── LanceDBBackend       ← all lancedb + pyarrow code; one backend per Collection
```

- **Collections** are the unit of schema isolation: each collection owns its own vector fields, property columns, and LanceDB table. Collections at the same URI never share state.
- `rrf_merge` is a module-level function in `fusion.py`, not a method on any class — it's a pure utility over `SearchResult` lists.

**Hard rule: `db.py` and `collection.py` must not import `pyarrow`, and only `db.py` may import `lancedb` (for the initial `connect`).** Every storage-engine specific call lives in `backend.py`. Swapping backends should be a single-file change.

## Architecture — big picture

**Properties vs. vectors are separate namespaces.** The API takes them as distinct arguments. Internally, vector columns are prefixed `__vec_<name>`; property names are rejected if they start with that prefix. Property names otherwise are stored as-is.

**Vector fields must be registered before use, per collection.** `collection.register_vector_field(name, dim)` adds a zero-copy `FixedSizeList<float32, dim>` column via LanceDB's `table.add_columns(pa.field(...))`. Registration is tracked in Arrow field metadata on each `__vec_<name>` column inside the Lance manifest (keys: `ovdb_dim`, `ovdb_description`, `ovdb_index`). Collection ownership is marked by an `ovdb_schema_version` sentinel on the `object_id` field. There is no on-disk state outside the Lance manifest.

**Schema grows automatically for properties.** Property columns are added on first write via Arrow-type inference from the sample value (`arrow_utils.python_value_to_arrow_type`). `None` alone cannot be inferred and raises `SchemaError`.

**Merge-update semantics.** `update()` and `batch_update()` only touch fields the caller passes. `None` clears a field on a specific object (the column stays, the cell becomes null). `update()` always raises `ObjectNotFound` when the target id is absent — use `upsert()` for insert-if-missing.

## Non-obvious LanceDB behaviors we encode

These are load-bearing — changing them quietly will break tests and correctness:

- **Null round-trip bugs (LanceDB #1325, #3105).** `table.update(values={col: None})` does not reliably round-trip. We use `values_sql={col: "CAST(NULL AS <sql_type>)"}` for scalar null-clearing and `merge_insert` with an Arrow-null FixedSizeList for vector null-clearing. The `arrow_type_to_sql_type` mapping lives in `arrow_utils.py`.
- **`batch_update` groups rows by column signature.** A single `merge_insert(...).when_matched_update_all()` call treats missing source columns as null overwrites. Without grouping, a row that specifies only `{"n": 1}` in the same batch as a sibling that sets `{"v": [...]}` would null-out `v` on the first row. See `_apply_update` and the grouping in `batch_update` in `backend.py`.
- **No auto-column-infer on `table.add`.** The backend always calls `add_columns(pa.field(...))` before an insert that references a new property or vector column.
- **No primary-key uniqueness at the storage layer.** Lance has no PK concept. `add()` uses `merge_insert.when_not_matched_insert_all` and raises `DuplicateObject` via `MergeResult.num_inserted_rows == 0`. `update()` uses `when_matched_update_all` and raises `ObjectNotFound` via `num_updated_rows == 0`. Distinct-id writes and update/upsert on existing rows are multi-writer safe; same-id concurrent inserts can still produce duplicate rows (commutative-append semantics) — see `docs/concurrency.md`.
- **`batch_update` rejects duplicate ids within a single batch.** `merge_insert` against multiple source rows sharing a key is implementation-defined in LanceDB, and column-signature grouping makes apply-order unreliable. `batch_update` raises `DuplicateObject` on intra-batch duplicates, mirroring `batch_add`. See `docs/concurrency.md` for the full multi-writer story.
- **Index metric wins at search time.** Once an index exists, LanceDB ignores the caller's `distance_type()`. We read the stored metric from Arrow field metadata (`ovdb_index`) and raise `MetricMismatch` if the caller passes a conflicting `metric=`. Do not soften this to a warning.
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
- `docs/concurrency.md` — multi-writer design decision, reference write-service architecture, and comparison with Weaviate / Qdrant / Milvus / Pinecone / pgvector.
- `docs/api.md` — full public API reference.
- `docs/filters.md` — DataFusion SQL `where=` syntax.
- `docs/testing.md` — what each test file covers and the v1 gate tests.

## Branch and PR conventions

- Develop on feature branches (current active branch: `claude/multimodal-object-store-ryKw0`). Never force-push to `main`.
- CI must be green before merging. The format check is part of lint.
- The commit trailer `https://claude.ai/code/session_...` is added automatically when Claude Code creates commits — leave it in place.

## Versioning and releases

[SemVer 2.0](https://semver.org/spec/v2.0.0.html) for the human-facing
version (`MAJOR.MINOR.PATCH` in `pyproject.toml`). Pre-release artifacts
produced from `main` use PEP 440 `0.X.Y.devN` (Python's accepted form).

Three CI workflows govern the release lifecycle:

1. **`.github/workflows/version-check.yml`** — runs on every PR targeting `main`. Fails if `pyproject.toml`'s version is not strictly greater (PEP 440 ordering) than `main`'s, *and* the PR touched `src/` or `pyproject.toml`. Pure-docs / pure-tests / pure-CI PRs are exempt.
2. **`.github/workflows/pre-release.yml`** — runs on every push to `main`. Builds an sdist + wheel with the version overridden to `<base>.dev<run_number>`, uploads them as workflow artifacts, and publishes a GitHub pre-release tagged `v<base>.dev<run_number>`. The override is build-time only; `pyproject.toml` on disk is unchanged.
3. **`.github/workflows/release.yml`** — runs when a `v*.*.*` tag is pushed. Verifies the tag matches `pyproject.toml`'s version exactly, builds, and creates a non-pre GitHub Release with auto-generated notes.

PyPI publishing is *not* wired up. The `release.yml` file contains a commented-out `pypa/gh-action-pypi-publish` step; enable it after configuring trusted publishing on PyPI (no API token needed).

Cutting a real release:

```bash
# 1. Bump the version in pyproject.toml on a PR (triggers version-check).
# 2. Merge the PR. Pre-release fires automatically.
# 3. From a clean main checkout, tag and push:
git tag v0.1.0
git push origin v0.1.0
# release.yml builds + publishes the GitHub Release.
```
