# Filter Syntax

The `where=` parameter on `search()`, `list()`, and `export_vectors()` is a
**DataFusion SQL** expression, evaluated server-side inside LanceDB. The
library does not introspect or validate the filter — it passes the string
through unchanged.

## Supported operators

### Comparison

```python
where="views > 1000"
where="score >= 0.5"
where="status = 'published'"
where="score BETWEEN 0.5 AND 0.9"
where="status != 'draft'"
```

### String matching

```python
where="title LIKE '%cat%'"                      # % = any sequence
where="title LIKE 'cat %'"                      # starts with "cat "
where="title ILIKE '%CAT%'"                     # case-insensitive
where="description IS NOT NULL"
where="description IS NULL"
```

### Logical

```python
where="views > 1000 AND tags LIKE '%cat%'"
where="status = 'draft' OR status = 'review'"
where="NOT (status = 'deleted')"
```

### Lists and arrays

```python
where="status IN ('published', 'review', 'featured')"
where="array_has(tags, 'cat')"                  # list membership
where="array_length(tags) > 0"
```

### Numeric / date helpers

DataFusion supports the standard SQL functions: `abs`, `round`, `floor`,
`ceil`, `power`, `least`, `greatest`, `coalesce`, and date/time functions
if your column types are dates.

## Examples in context

Find the top 10 cat videos:

```python
store.search(
    query_vector=[...],
    vector_field="text_openai",
    limit=10,
    where="array_has(tags, 'cat') AND views > 1000",
)
```

Export all embeddings for objects reviewed after a cutoff:

```python
ids, arr = store.export_vectors(
    "text_openai",
    where="review_score >= 0.8 AND status = 'published'",
)
```

Paginate through a subset:

```python
page = store.list_objects(
    where="views > 100",
    select=["title", "views"],
    limit=50,
    offset=100,
)
```

## Escaping caller-supplied input

The library does not sanitize filter strings. If any part of `where` is
composed from untrusted input, the caller must escape it themselves. For
string literals, single-quote with doubled embedded quotes:

```python
def quote_literal(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"

user_title = "that's a cat"                     # untrusted
where = f"title = {quote_literal(user_title)}"  # safe: title = 'that''s a cat'
```

Prefer composing filters from known-safe parameters (numeric comparisons,
enum-constrained statuses) rather than concatenating arbitrary text.

## Column names in filters

Use the public property name directly. Vector columns are stored with the
`__vec_` prefix internally, but they are not meaningful in filter
expressions — use scalar properties to narrow candidates.

```python
where="title LIKE '%cat%'"            # OK — title is a property column
# where="__vec_text_openai IS NOT NULL"   # works but not useful
```

## Backend lock-in

Filter syntax is LanceDB / DataFusion-specific by design. If the backend is
ever swapped (see [architecture.md](architecture.md#module-layout)), filter
strings are the only user-facing surface that may need revisiting. This is an
accepted trade-off: building a backend-agnostic filter DSL would add
significant surface area for minimal real benefit.
