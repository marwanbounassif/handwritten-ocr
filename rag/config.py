"""Centralized configuration loaded from .env file.

All other modules import settings from here — never call os.getenv directly elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Walk up from this file to find the project root .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables."""

    QDRANT_URL: str
    QDRANT_API_KEY: str
    COLLECTION_NAME: str
    CONFIDENCE_WRITE_BACK_THRESHOLD: float
    TOP_K_DEFAULT: int
    HF_HOME: str


def _load_settings() -> Settings:
    qdrant_url = os.getenv("QDRANT_END_POINT") or os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise EnvironmentError(
            "Missing required environment variables. "
            "Please set QDRANT_END_POINT (or QDRANT_URL) and QDRANT_API_KEY in your .env file.\n"
            "Get these from https://cloud.qdrant.io → Clusters → your cluster → API Keys."
        )

    return Settings(
        QDRANT_URL=qdrant_url,
        QDRANT_API_KEY=qdrant_api_key,
        COLLECTION_NAME=os.getenv("COLLECTION_NAME", "handwritten_notes"),
        CONFIDENCE_WRITE_BACK_THRESHOLD=float(
            os.getenv("CONFIDENCE_WRITE_BACK_THRESHOLD", "0.85")
        ),
        TOP_K_DEFAULT=int(os.getenv("TOP_K_DEFAULT", "5")),
        HF_HOME=os.getenv("HF_HOME", "~/.cache/huggingface"),
    )


settings = _load_settings()
