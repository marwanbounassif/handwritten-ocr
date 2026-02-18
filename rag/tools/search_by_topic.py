"""Tool: search_by_topic — retrieve notes related to an inferred topic."""

from __future__ import annotations

from loguru import logger
from rag.client import get_client
from rag.config import settings
from rag.embeddings.bge_embedder import embed_dense
from rag.tools.search_context import SearchResult


def search_by_topic(
    inferred_topic: str,
    top_k: int = 10,
    return_unique_tags_only: bool = False,
) -> list[SearchResult] | list[str]:
    """Search for notes related to an inferred topic using dense embeddings only.

    Broad topic queries don't benefit from sparse matching, so this tool
    uses only the text_dense vector.

    Args:
        inferred_topic: A natural language description of the topic to search for.
        top_k: Maximum number of results to return.
        return_unique_tags_only: If True, return a sorted list of unique topic_tags
            across all results instead of full SearchResult objects.

    Returns:
        List of SearchResult dataclasses, or list of unique tag strings if
        return_unique_tags_only is True.
    """
    client = get_client()
    dense_vecs = embed_dense([inferred_topic])

    from qdrant_client.models import FieldCondition, Filter, Range

    results = client.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=dense_vecs[0],
        using="text_dense",
        limit=top_k,
        with_payload=True,
        query_filter=Filter(must=[
            FieldCondition(key="confidence_score", range=Range(gt=0.0)),
        ]),
    )

    logger.debug(
        "search_by_topic returned {} results for topic='{}'",
        len(results.points),
        inferred_topic,
    )

    if return_unique_tags_only:
        all_tags: set[str] = set()
        for point in results.points:
            tags = point.payload.get("topic_tags", [])
            all_tags.update(tags)
        return sorted(all_tags)

    return [
        SearchResult(
            confirmed_text=point.payload.get("confirmed_text", ""),
            raw_ocr_text=point.payload.get("raw_ocr_text", ""),
            score=point.score,
            source_image_id=point.payload.get("source_image_id", ""),
            topic_tags=point.payload.get("topic_tags", []),
            chunk_type=point.payload.get("chunk_type", ""),
            confidence_score=point.payload.get("confidence_score", 0.0),
        )
        for point in results.points
    ]
