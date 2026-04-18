# `object_store` Documentation

Welcome. This folder documents the design and public API of the `object_store`
Python library — a thin, object-centric layer on top of LanceDB for managing
multimodal objects with evolving annotations and multiple named embedding
vectors per object.

## Reading order

1. **[concepts.md](concepts.md)** — the core mental model: objects, properties,
   vectors, and why vector fields are explicitly registered.
2. **[architecture.md](architecture.md)** — the technical design: module layout,
   the registry, LanceDB API usage, column naming, schema evolution, score
   conversion, null-clearing strategy, concurrency.
3. **[api.md](api.md)** — full reference for every public method on
   `ObjectStore`, every dataclass, and every exception.
4. **[filters.md](filters.md)** — the `where` filter syntax (DataFusion SQL)
   with practical examples.
5. **[testing.md](testing.md)** — what the test suite covers and how to run
   it.

## Quick links

- [API: `ObjectStore.add`](api.md#add)
- [API: `ObjectStore.update`](api.md#update)
- [API: `ObjectStore.search`](api.md#search)
- [Architecture: score conversion](architecture.md#score-conversion)
- [Architecture: null-clearing](architecture.md#null-clearing)
- [Concepts: why register vector fields?](concepts.md#why-register-vector-fields)
