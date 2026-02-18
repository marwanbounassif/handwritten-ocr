"""Ingestion: upsert confirmed transcriptions into Qdrant."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from loguru import logger
from qdrant_client.models import PointStruct

from rag.client import get_client
from rag.config import settings
from rag.embeddings.bge_embedder import embed_dense, embed_sparse
from rag.embeddings.visual_embedder import embed_image_from_base64, embed_text_clip

# Namespace UUID for deterministic ID generation
_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def upsert_transcription(
    confirmed_text: str,
    raw_ocr_text: str,
    source_image_id: str,
    region_coords: dict,
    confidence_score: float,
    topic_tags: list[str],
    chunk_type: str,
    image_crop_base64: str | None = None,
    from_human_review: bool = False,
    respect_confidence_threshold: bool = True,
) -> str | None:
    """Upsert a confirmed transcription into the handwritten_notes collection.

    Generates a deterministic UUID5 from source_image_id and region_coords for
    idempotent upserts. Embeds text with BGE-M3 (dense + sparse) and generates
    a visual vector from the image crop or CLIP text fallback.

    Args:
        confirmed_text: The confirmed/corrected transcription text.
        raw_ocr_text: The original raw OCR output.
        source_image_id: Identifier of the source image.
        region_coords: Dict with keys x, y, w, h defining the text region.
        confidence_score: Confidence score of the transcription (0.0-1.0).
        topic_tags: List of topic tags for this chunk.
        chunk_type: One of "sentence", "phrase", or "word".
        image_crop_base64: Optional base64-encoded image crop of the region.
        from_human_review: Whether this transcription was human-verified.
        respect_confidence_threshold: If True, skip upsert when confidence is
            below CONFIDENCE_WRITE_BACK_THRESHOLD.

    Returns:
        The UUID string of the upserted point, or None if skipped due to
        low confidence.
    """
    if respect_confidence_threshold and confidence_score < settings.CONFIDENCE_WRITE_BACK_THRESHOLD:
        logger.warning(
            "Skipping upsert: confidence {:.3f} < threshold {:.3f} for '{}'",
            confidence_score,
            settings.CONFIDENCE_WRITE_BACK_THRESHOLD,
            confirmed_text[:50],
        )
        return None

    # Deterministic UUID for idempotent upserts
    point_id = str(
        uuid.uuid5(_NAMESPACE, f"{source_image_id}_{region_coords}")
    )

    # Embed text
    dense_vecs = embed_dense([confirmed_text])
    sparse_vecs = embed_sparse([confirmed_text])

    # Visual vector: image crop if available, else CLIP text fallback
    if image_crop_base64:
        visual_vec = embed_image_from_base64(image_crop_base64)
    else:
        visual_vec = embed_text_clip(confirmed_text)

    payload = {
        "confirmed_text": confirmed_text,
        "raw_ocr_text": raw_ocr_text,
        "source_image_id": source_image_id,
        "region_coords": region_coords,
        "confidence_score": confidence_score,
        "topic_tags": topic_tags,
        "chunk_type": chunk_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "from_human_review": from_human_review,
    }

    point = PointStruct(
        id=point_id,
        vector={
            "text_dense": dense_vecs[0],
            "text_sparse": sparse_vecs[0],
            "visual": visual_vec,
        },
        payload=payload,
    )

    client = get_client()
    client.upsert(
        collection_name=settings.COLLECTION_NAME,
        points=[point],
    )

    logger.info("Upserted point {} for '{}'", point_id, confirmed_text[:50])
    return point_id
