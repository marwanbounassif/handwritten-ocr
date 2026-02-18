"""S3 storage operations for handwritten page images.

Provides upload, download, listing, and sync functions.
AWS credentials are read from environment variables by boto3 automatically.
Bucket name and prefix are read from S3_BUCKET_NAME and S3_PREFIX env vars.
"""

from __future__ import annotations

import os
from pathlib import Path

import boto3
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_LOCAL_DIR = _PROJECT_ROOT / "data" / "input"

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def _bucket_name() -> str:
    """Return the S3 bucket name, stripping ARN prefix if present."""
    raw = os.environ.get("S3_BUCKET_NAME", "")
    if not raw:
        raise EnvironmentError(
            "S3_BUCKET_NAME not set in .env. "
            "Set it to your bucket name (e.g., 'handwritten-ocr')."
        )
    # Handle ARN format: arn:aws:s3:::bucket-name → bucket-name
    if raw.startswith("arn:aws:s3:::"):
        return raw.split(":::")[-1]
    return raw


def _prefix() -> str:
    """Return the S3 key prefix (subfolder), with trailing slash."""
    prefix = os.environ.get("S3_PREFIX", "")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


def _client():
    """Return a boto3 S3 client."""
    return boto3.client("s3")


def upload_image(local_path: str | Path, s3_key: str | None = None) -> str:
    """Upload a local image to S3.

    Args:
        local_path: Path to the local image file.
        s3_key: S3 object key. Defaults to prefix + filename.

    Returns:
        The S3 key of the uploaded object.
    """
    local_path = Path(local_path)
    if s3_key is None:
        s3_key = _prefix() + local_path.name

    _client().upload_file(str(local_path), _bucket_name(), s3_key)
    logger.info("Uploaded {} → s3://{}/{}", local_path.name, _bucket_name(), s3_key)
    return s3_key


def download_image(
    s3_key: str, local_dir: str | Path | None = None
) -> Path:
    """Download an image from S3 to local cache.

    Skips download if the file already exists locally.

    Args:
        s3_key: S3 object key.
        local_dir: Local directory to save to. Defaults to data/input/.

    Returns:
        Path to the local file.
    """
    local_dir = Path(local_dir) if local_dir else _DEFAULT_LOCAL_DIR
    local_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(s3_key).name
    local_path = local_dir / filename

    if local_path.exists():
        logger.debug("Already cached: {}", local_path)
        return local_path

    _client().download_file(_bucket_name(), s3_key, str(local_path))
    logger.info("Downloaded s3://{}/{} → {}", _bucket_name(), s3_key, local_path)
    return local_path


def list_images(prefix: str | None = None) -> list[str]:
    """List all image keys in the S3 bucket.

    Args:
        prefix: Override the default S3_PREFIX. If None, uses env var.

    Returns:
        List of S3 object keys for image files.
    """
    pfx = prefix if prefix is not None else _prefix()
    client = _client()
    bucket = _bucket_name()

    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=pfx):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if Path(key).suffix.lower() in _IMAGE_EXTENSIONS:
                keys.append(key)

    logger.info("Found {} images in s3://{}/{}", len(keys), bucket, pfx)
    return keys


def sync_all(local_dir: str | Path | None = None) -> list[Path]:
    """Download all S3 images not already in local cache.

    Args:
        local_dir: Local directory to sync to. Defaults to data/input/.

    Returns:
        List of local paths for all synced images.
    """
    keys = list_images()
    paths = []
    for key in keys:
        path = download_image(key, local_dir)
        paths.append(path)
    logger.info("Synced {} images from S3.", len(paths))
    return paths
