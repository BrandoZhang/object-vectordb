# Performance benchmarks

This directory holds opt-in micro-benchmarks for the hot paths listed in the
spec. They are **not** part of the default pytest run and **not** enforced in
CI — benchmarks are flaky across hardware, and gating PRs on absolute numbers
produces noise, not signal.

## How to run

```bash
uv sync --all-extras                                  # installs pytest-benchmark

# Quick tier (~30 s; 1 K-row datasets)
uv run pytest benches/ --benchmark-only -m "not full"

# Full tier (~5–10 min; 100 K–1 M-row datasets; matches the spec's target scales)
uv run pytest benches/ --benchmark-only -m full

# Both tiers
uv run pytest benches/ --benchmark-only
```

Dump a JSON report for offline comparison:

```bash
uv run pytest benches/ --benchmark-only --benchmark-json=bench.json
```

## Spec targets

| Operation                                    | Target  | Data size | Bench file                |
| -------------------------------------------- | ------- | --------- | ------------------------- |
| `add` + `get` roundtrip                      | < 10 ms | single    | `test_add_get.py`         |
| `search` on indexed vector field             | < 50 ms | 100 K     | `test_search.py` (`full`) |
| `batch_update`                               | < 5 s   | 1 000     | `test_batch.py`           |
| `export_vectors`                             | < 10 s  | 100 K     | `test_export.py` (`full`) |
| `register_vector_field` (zero-copy)          | < 1 s   | 1 M rows  | `test_register.py` (`full`) |

Spec targets are treated as **targets, not gates**. Missed numbers are
recorded honestly and investigated in a follow-up; they do not block a commit.

## Design notes

- **Runner**: `pytest-benchmark`. Uses `benchmark.pedantic(fn, rounds=N, iterations=1)`
  for ops where setup is expensive; plain `benchmark(fn)` for cheap ops.
- **Vector dimension**: 128 by default (`DEFAULT_DIM` in `conftest.py`).
- **Seeding**: the `seeded_collection_factory` fixture caches datasets per
  `(size, dim, with_index)` tuple across a pytest session. Without caching,
  the 100 K-row tier would re-seed on every bench round.
- **Reproducibility**: a fixed numpy seed (42) is used for all random
  vectors. IVF_PQ clustering is randomized internally by LanceDB; accept that
  as jitter.

## Recording observed numbers

After running the full tier, paste the median values into the
**Performance** table in the top-level `README.md`. Include the hardware
you measured on (CPU / RAM / storage) so readers can calibrate.
