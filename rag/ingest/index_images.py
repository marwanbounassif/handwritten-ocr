"""Auto-index pipeline: preprocess images, crop, and upsert visual embeddings to Qdrant.

Indexes image crops with CLIP visual embeddings only (no text/OCR).
Text embeddings are added later via the annotation notebook.
"""

from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path

from loguru import logger
from PIL import Image
from qdrant_client.models import PointStruct, SparseVector

from rag.client import get_client
from rag.config import settings
from rag.embeddings.visual_embedder import embed_image
from rag.schema import ensure_collection_exists

# Deterministic UUID namespace — same as upsert.py for consistency
_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# Cropping defaults
NUM_ROWS = 6
OVERLAP_PX = 20
MIN_CROP_HEIGHT = 80

# Zero vectors for placeholder text embeddings
_ZERO_DENSE = [0.0] * 1024
_ZERO_SPARSE = SparseVector(indices=[], values=[])


def _make_row_crops(
    img: Image.Image,
    num_rows: int = NUM_ROWS,
    overlap_px: int = OVERLAP_PX,
) -> list[tuple[Image.Image, dict]]:
    """Divide an image into horizontal strips.

    Args:
        img: Source PIL image.
        num_rows: Number of horizontal strips.
        overlap_px: Pixel overlap between adjacent strips.

    Returns:
        List of (crop_image, region_coords_dict) tuples.
        region_coords has keys: x, y, w, h.
    """
    w, h = img.size
    base_height = h // num_rows
    crops = []
    for i in range(num_rows):
        y_start = max(0, i * base_height - overlap_px)
        y_end = min(h, (i + 1) * base_height + overlap_px)
        if y_end - y_start < MIN_CROP_HEIGHT:
            continue
        crop = img.crop((0, y_start, w, y_end))
        region = {"x": 0, "y": y_start, "w": w, "h": y_end - y_start}
        crops.append((crop, region))
    return crops


def _crop_to_base64(crop: Image.Image) -> str:
    """Encode a PIL image crop as base64 PNG."""
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _point_id(source_image_id: str, region_coords: dict) -> str:
    """Generate deterministic UUID5 matching upsert.py convention."""
    return str(uuid.uuid5(_NAMESPACE, f"{source_image_id}_{region_coords}"))


def index_image(
    image_path: str | Path,
    num_rows: int = NUM_ROWS,
    overlap_px: int = OVERLAP_PX,
    preprocessing: list[str] | None = None,
) -> int:
    """Index a single image: preprocess, crop, CLIP embed, upsert to Qdrant.

    Each crop is upserted with a visual embedding and empty text fields.
    Text embeddings are added later during manual annotation.

    Args:
        image_path: Path to the source image.
        num_rows: Number of horizontal strips to divide the image into.
        overlap_px: Pixel overlap between strips.
        preprocessing: List of preprocessing transforms to apply.
            Defaults to ["deskew", "high_contrast"].

    Returns:
        Number of crops indexed.
    """
    from ocr_agent.tools import preprocess_image

    image_path = Path(image_path)
    source_id = image_path.stem

    if preprocessing is None:
        preprocessing = ["deskew", "high_contrast"]

    # Preprocess — returns path to temp file
    processed_path = preprocess_image(str(image_path), preprocessing)
    processed_img = Image.open(processed_path).convert("RGB")

    crops = _make_row_crops(processed_img, num_rows, overlap_px)
    if not crops:
        logger.warning("No crops generated for {}", image_path.name)
        return 0

    client = get_client()
    ensure_collection_exists(client)

    points = []
    for crop_img, region in crops:
        pid = _point_id(source_id, region)
        visual_vec = embed_image(crop_img)
        b64 = _crop_to_base64(crop_img)

        point = PointStruct(
            id=pid,
            vector={
                "text_dense": _ZERO_DENSE,
                "text_sparse": _ZERO_SPARSE,
                "visual": visual_vec,
            },
            payload={
                "confirmed_text": "",
                "raw_ocr_text": "",
                "source_image_id": source_id,
                "region_coords": region,
                "confidence_score": 0.0,
                "topic_tags": [],
                "chunk_type": "phrase",
                "from_human_review": False,
                "image_crop_base64": b64,
            },
        )
        points.append(point)

    client.upsert(collection_name=settings.COLLECTION_NAME, points=points)
    logger.info(
        "Indexed {} crops from {} into '{}'",
        len(points),
        image_path.name,
        settings.COLLECTION_NAME,
    )
    return len(points)


def index_all(
    local_dir: str | Path | None = None,
    num_rows: int = NUM_ROWS,
    overlap_px: int = OVERLAP_PX,
    sync_s3: bool = True,
) -> int:
    """Index all images: optionally sync from S3, then index each.

    Idempotent — re-indexing the same image overwrites the same Qdrant points
    thanks to deterministic UUIDs.

    Args:
        local_dir: Directory containing images. Defaults to data/input/.
        num_rows: Number of horizontal strips per image.
        overlap_px: Pixel overlap between strips.
        sync_s3: If True, sync images from S3 before indexing.

    Returns:
        Total number of crops indexed.
    """
    if local_dir is None:
        local_dir = Path(__file__).resolve().parent.parent.parent / "data" / "input"
    local_dir = Path(local_dir)

    if sync_s3:
        from rag.storage.s3 import sync_all
        sync_all(local_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
    images = sorted(
        f for f in local_dir.iterdir()
        if f.suffix.lower() in image_extensions
    )

    if not images:
        logger.warning("No images found in {}", local_dir)
        return 0

    logger.info("Indexing {} images from {}...", len(images), local_dir)
    total = 0
    for img_path in images:
        count = index_image(img_path, num_rows, overlap_px)
        total += count

    logger.info("Done. Indexed {} total crops from {} images.", total, len(images))
    return total
