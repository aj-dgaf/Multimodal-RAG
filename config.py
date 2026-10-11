"""
config.py — Central configuration

Every setting (paths, model names, thresholds, chunk sizes, device) lives here.
No magic numbers elsewhere in the codebase.

Not run directly; imported by all other modules.
"""

import torch
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"
IMAGE_DIR = DATA_DIR / "images"
CACHE_PATH = DATA_DIR / "cache.json"
RESULTS_PATH = DATA_DIR / "results.json"

# Models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Retrieval & Chunking
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
TOP_K_TEXT = 5
TOP_K_IMAGES = 3
MAX_IMAGES = 4
TEXT_CANDIDATES = 40

# Image Extraction
MIN_IMAGE_SIZE = 150
MAX_IMAGE_EDGE = 1024
LOGO_REPEAT_THRESHOLD = 3
VECTOR_DRAWING_THRESHOLD = 50

# Score Thresholds (Placeholders)
TEXT_SCORE_THRESHOLD = 0.5
IMAGE_SCORE_THRESHOLD = 0.2

# Retry Settings
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2

# Evaluation pacing: seconds before each Gemini call (free tier allows 15 requests/minute)
EVAL_CALL_DELAY = 5
