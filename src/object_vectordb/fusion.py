"""Rank-fusion utilities.

Pure-Python helpers that combine multiple `SearchResult` lists. No LanceDB or
pyarrow involvement — safe to call from any context.
"""

from __future__ import annotations

from typing import Any

from .types import SearchResult


def rrf_merge(
    *result_lists: list[SearchResult],
    k: int = 60,
    limit: int | None = None,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion over multiple SearchResult lists.

    For each object id, the fused score is sum over input lists of
    ``1 / (k + rank)``, where ``rank`` is the 1-indexed position of the
    object in that list. Objects present in only some lists still receive
    a score (contributions from absent lists are zero).

    Ties break by first-seen order across the concatenated inputs, so the
    output is deterministic for a given input order.

    Properties on each returned `SearchResult` come from the first occurrence
    of that object id across the inputs.
    """
    scores: dict[str, float] = {}
    props: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    counter = 0
    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            scores[hit.object_id] = scores.get(hit.object_id, 0.0) + 1.0 / (k + rank)
            if hit.object_id not in props:
                props[hit.object_id] = hit.properties
                first_seen[hit.object_id] = counter
                counter += 1
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], first_seen[kv[0]]),
    )
    if limit is not None:
        ranked = ranked[:limit]
    return [
        SearchResult(object_id=oid, score=score, properties=props[oid]) for oid, score in ranked
    ]
