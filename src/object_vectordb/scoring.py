"""Convert LanceDB's `_distance` (lower=better) to a similarity score (higher=better)."""

from __future__ import annotations

SUPPORTED_METRICS = frozenset({"cosine", "l2", "dot"})


def normalize_metric(metric: str) -> str:
    m = metric.lower()
    if m == "euclidean":
        m = "l2"
    if m not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric {metric!r}. Supported: {sorted(SUPPORTED_METRICS)}")
    return m


def distance_to_score(distance: float, metric: str) -> float:
    """Convert a LanceDB `_distance` value to a similarity score.

    Formulas (verified against LanceDB 0.30.x):
      cosine: LanceDB distance = 1 - cos_sim   -> score = 1 - distance   (range [-1, 1])
      dot:    LanceDB distance = 1 - dot(q,v)  -> score = 1 - distance   (range unbounded)
      l2:     LanceDB distance = ||q-v||^2     -> score = 1 / (1 + d)    (range (0, 1]; monotonic only)
    """
    m = normalize_metric(metric)
    if m == "cosine":
        return 1.0 - float(distance)
    if m == "dot":
        return 1.0 - float(distance)
    # l2
    return 1.0 / (1.0 + float(distance))
