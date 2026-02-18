"""Scoring utilities for result fusion."""

from __future__ import annotations


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Standard RRF: score = sum(1 / (k + rank)) across all lists,
    where rank is 1-indexed.

    Args:
        result_lists: List of ranked result lists. Each inner list contains
            (point_id, score) tuples sorted by score descending.
        k: RRF constant (default 60). Higher values reduce the impact of
            high-ranking results.

    Returns:
        List of (point_id, fused_score) tuples sorted by fused score descending.
    """
    scores: dict[str, float] = {}
    for result_list in result_lists:
        for rank, (point_id, _score) in enumerate(result_list, start=1):
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
