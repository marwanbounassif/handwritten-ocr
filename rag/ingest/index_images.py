"""Auto-index pipeline: preprocess images, crop, and upsert visual embeddings to Qdrant.

Indexes image crops with CLIP visual embeddings only (no text/OCR).
Text embeddings are added later via the annotation notebook.
"""

from __future__ import annotations

import base64
import hashlib
import io
import uuid
from pathlib import Path

from loguru import logger
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()  # enables Image.open() on .heic/.heif files
from qdrant_client.models import PointStruct, SparseVector

from rag.client import get_client
from rag.config import settings
from rag.embeddings.visual_embedder import embed_image
from rag.schema import ensure_collection_exists

# Deterministic UUID namespace — same as upsert.py for consistency
_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# Crop defaults
NUM_CROPS = 12
MIN_CROP_SIZE = 128

# Scale range as fraction of the shorter image dimension
SCALE_MIN = 0.15
SCALE_MAX = 0.45

# Zero vectors for placeholder text embeddings
_ZERO_DENSE = [0.0] * 1024
_ZERO_SPARSE = SparseVector(indices=[], values=[])


def _deterministic_random(source_image_id: str, num_crops: int, img_w: int, img_h: int):
    """Generate deterministic pseudo-random crop parameters from image id.

    Uses a hash-seeded RNG so the same image always produces the same crops,
    making the pipeline idempotent without needing to track state.
    """
    import random

    seed = int(hashlib.sha256(source_image_id.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    short_side = min(img_w, img_h)
    crops = []
    for _ in range(num_crops):
        scale = rng.uniform(SCALE_MIN, SCALE_MAX)
        crop_w = max(MIN_CROP_SIZE, int(img_w * scale))
        crop_h = max(MIN_CROP_SIZE, int(img_h * scale))

        # Clamp to image bounds
        crop_w = min(crop_w, img_w)
        crop_h = min(crop_h, img_h)

        x = rng.randint(0, img_w - crop_w)
        y = rng.randint(0, img_h - crop_h)
        crops.append({"x": x, "y": y, "w": crop_w, "h": crop_h})

    return crops


def make_random_crops(
    img: Image.Image,
    source_image_id: str,
    num_crops: int = NUM_CROPS,
) -> list[tuple[Image.Image, dict]]:
    """Generate pseudo-random crops from an image.

    Crops are deterministic per source_image_id so re-running produces
    identical results. Varies both position and scale.

    Args:
        img: Source PIL image.
        source_image_id: Used to seed the RNG for reproducibility.
        num_crops: Number of crops to generate.

    Returns:
        List of (crop_image, region_coords_dict) tuples.
    """
    w, h = img.size
    regions = _deterministic_random(source_image_id, num_crops, w, h)

    results = []
    for region in regions:
        x, y, cw, ch = region["x"], region["y"], region["w"], region["h"]
        crop = img.crop((x, y, x + cw, y + ch))
        results.append((crop, region))

    return results


def _point_id(source_image_id: str, region_coords: dict) -> str:
    """Generate deterministic UUID5 matching upsert.py convention."""
    return str(uuid.uuid5(_NAMESPACE, f"{source_image_id}_{region_coords}"))


def index_image(
    image_path: str | Path,
    num_crops: int = 0,
    preprocessing: list[str] | None = None,
) -> int:
    """Index a single image: preprocess, optionally crop, CLIP embed, upsert to Qdrant.

    Each point is upserted with a visual embedding and empty text fields.
    Text embeddings are added later during manual annotation.

    Args:
        image_path: Path to the source image.
        num_crops: Number of random crops to generate. 0 (default) indexes
            the full image as a single point with no cropping.
        preprocessing: List of preprocessing transforms to apply.
            Defaults to ["deskew", "high_contrast", "binarize"].

    Returns:
        Number of points indexed.
    """
    from ocr_agent.tools import preprocess_image

    image_path = Path(image_path)
    source_id = image_path.stem

    if preprocessing is None:
        preprocessing = ["deskew", "high_contrast", "binarize"]

    # Preprocess — returns path to temp file
    processed_path = preprocess_image(str(image_path), preprocessing)
    processed_img = Image.open(processed_path)

    if num_crops > 0:
        crops = make_random_crops(processed_img, source_id, num_crops)
    else:
        # Index the full image as a single point
        w, h = processed_img.size
        region = {"x": 0, "y": 0, "w": w, "h": h}
        crops = [(processed_img, region)]

    if not crops:
        logger.warning("No crops generated for {}", image_path.name)
        return 0

    client = get_client()
    ensure_collection_exists(client)

    points = []
    for crop_img, region in crops:
        pid = _point_id(source_id, region)
        visual_vec = embed_image(crop_img)

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
            },
        )
        points.append(point)

    client.upsert(collection_name=settings.COLLECTION_NAME, points=points)
    label = f"{len(points)} crops" if num_crops > 0 else "full image"
    logger.info("Indexed {} from {} into '{}'", label, image_path.name, settings.COLLECTION_NAME)
    return len(points)


def index_all(
    local_dir: str | Path | None = None,
    num_crops: int = 0,
    sync_s3: bool = True,
) -> int:
    """Index all images: optionally sync from S3, then index each.

    Idempotent — re-indexing the same image overwrites the same Qdrant points
    thanks to deterministic UUIDs seeded from the image filename.

    Args:
        local_dir: Directory containing images. Defaults to data/input/.
        num_crops: Number of random crops per image. 0 (default) indexes
            the full image as a single point with no cropping.
        sync_s3: If True, sync images from S3 before indexing.

    Returns:
        Total number of points indexed.
    """
    if local_dir is None:
        local_dir = Path(__file__).resolve().parent.parent.parent / "data" / "input"
    local_dir = Path(local_dir)

    if sync_s3:
        from rag.storage.s3 import sync_all
        sync_all(local_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".heic", ".heif"}
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
        count = index_image(img_path, num_crops)
        total += count

    logger.info("Done. Indexed {} total points from {} images.", total, len(images))
    return total
