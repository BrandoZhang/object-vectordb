from __future__ import annotations

from object_vectordb import ObjectVectorDB, SearchResult


def _mk(oid: str, score: float, props: dict | None = None) -> SearchResult:
    return SearchResult(object_id=oid, score=score, properties=props or {})


def test_rrf_merge_basic_fusion():
    a = [_mk("x", 1.0), _mk("y", 0.9), _mk("z", 0.8)]
    b = [_mk("y", 1.0), _mk("z", 0.9), _mk("w", 0.8)]
    fused = ObjectVectorDB.rrf_merge(a, b, k=60)
    ids = [r.object_id for r in fused]
    # y appears in both lists at good ranks and should win
    assert ids[0] == "y"
    assert "x" in ids and "z" in ids and "w" in ids


def test_rrf_merge_respects_limit():
    a = [_mk("a", 1.0), _mk("b", 0.9), _mk("c", 0.8)]
    b = [_mk("b", 1.0), _mk("c", 0.9), _mk("d", 0.8)]
    fused = ObjectVectorDB.rrf_merge(a, b, k=60, limit=2)
    assert len(fused) == 2


def test_rrf_merge_k_parameter_affects_scores():
    a = [_mk("x", 1.0), _mk("y", 0.5)]
    b = [_mk("y", 1.0), _mk("x", 0.5)]
    small_k = ObjectVectorDB.rrf_merge(a, b, k=1)
    large_k = ObjectVectorDB.rrf_merge(a, b, k=1000)
    # Small k → scores more spread. With k=1, each rank-1 contributes 1/2, rank-2 contributes 1/3.
    # Both x and y have identical profiles, so scores are equal.
    assert small_k[0].score != large_k[0].score
    # Stable tie-breaking by first-seen order
    assert {r.object_id for r in small_k[:2]} == {"x", "y"}


def test_rrf_merge_preserves_properties_from_first_occurrence():
    a = [_mk("x", 1.0, {"source": "A"})]
    b = [_mk("x", 1.0, {"source": "B"})]
    fused = ObjectVectorDB.rrf_merge(a, b)
    assert fused[0].properties == {"source": "A"}


def test_rrf_merge_empty_lists():
    fused = ObjectVectorDB.rrf_merge([], [])
    assert fused == []


def test_rrf_merge_single_list():
    a = [_mk("a", 1.0), _mk("b", 0.5)]
    fused = ObjectVectorDB.rrf_merge(a)
    assert [r.object_id for r in fused] == ["a", "b"]
