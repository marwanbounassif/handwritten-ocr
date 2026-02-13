"""
Configuration for the agentic OCR pipeline.
All tuneable parameters live here.
"""

# ── LLM Configuration ─────────────────────────────────────────────
OPENAI_BASE_URL = "http://localhost:11434/v1"
OPENAI_API_KEY = "ollama"
OPENAI_MODEL = "qwen3:32b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 4096
LLM_TIMEOUT = 300  # seconds
LLM_STREAM = True             # stream responses (shows live output, avoids timeouts on slow models)
LLM_ENABLE_THINKING = True    # Qwen3 thinking mode — model reasons in <think> tags before answering
LLM_STOP_SEQUENCES = []       # optional stop sequences, e.g. ["\n\n", "IN:"]

# ── OCR Model Configuration ──────────────────────────────────────
OLMOCR_MODEL = "allenai/olmOCR-2-7B-1025"
OCR_MAX_PIXELS = 1024 * 1024  # ~1 megapixel, safe for 48GB MPS
OCR_MIN_PIXELS = 256 * 256
OCR_MAX_NEW_TOKENS = 2048
OCR_PROMPT = "Extract and return all the text from this handwritten document."

# ── Orchestrator Configuration ────────────────────────────────────
MAX_ITERATIONS = 10
ACCEPT_THRESHOLD = 85           # critic confidence to auto-accept
PLATEAU_PATIENCE = 2            # stop if no improvement for N iterations
AGREEMENT_THRESHOLD = 80        # % agreement below which a tiebreaker OCR pass runs
# Each entry is either a single transform string or a list of transforms applied in order.
# The orchestrator tries them top-to-bottom: first two as the initial reads, then extras on re-OCR.
PREPROCESSING_STRATEGIES = [
    "original",
    ["deskew", "high_contrast", "sharpen"],           # best general pipeline for phone photos
    ["deskew", "denoise", "high_contrast"],            # noisy / low-light photos
    ["deskew", "remove_lines", "high_contrast"],       # lined notebook paper
    ["deskew", "high_contrast", "binarize"],           # very faint ink
]
