"""
retriever.py — Phase 1: Text-only retrieval

Loads the FAISS text index and metadata built by ingest.py, embeds
the user's query with bge-small (with retrieval instruction prefix),
and returns the top-k most relevant text chunks with scores.

Phase 3 will extend this to support "text+tables" and "multimodal" modes.

Usage:
    Imported by generator.py and app.py
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config

# bge-small expects this prefix on queries (but NOT on documents)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Module-level cache so models/indexes are loaded only once
_text_index = None
_text_meta = None
_text_model = None


# --- 1. Load index and model ---

def _load_text_index() -> tuple:
    """Load the FAISS text index and metadata from disk."""
    global _text_index, _text_meta
    if _text_index is not None:
        return _text_index, _text_meta

    index_path = config.INDEX_DIR / "text.index"
    meta_path = config.INDEX_DIR / "text_meta.json"

    if not index_path.exists() or not meta_path.exists():
        print(f"ERROR: Index not found at {config.INDEX_DIR}. Run ingest.py first.")
        raise FileNotFoundError(f"Index not found at {config.INDEX_DIR}")

    _text_index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        _text_meta = json.load(f)

    print(f"Loaded text index: {_text_index.ntotal} vectors")
    return _text_index, _text_meta


def _load_text_model() -> SentenceTransformer:
    """Load the bge-small sentence-transformer model."""
    global _text_model
    if _text_model is not None:
        return _text_model

    _text_model = SentenceTransformer(config.TEXT_EMBED_MODEL, device=config.DEVICE)
    return _text_model


# --- 2. Search ---

def search(question: str, mode: str = "text", top_k: int | None = None) -> dict:
    """Retrieve top-k evidence for a question. Returns a result dict."""
    if top_k is None:
        top_k = config.TOP_K_TEXT

    index, meta = _load_text_index()
    model = _load_text_model()

    # Embed query with bge retrieval prefix
    query_text = BGE_QUERY_PREFIX + question
    query_vec = model.encode([query_text], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)

    # Search more candidates then filter by mode
    n_search = min(config.TEXT_CANDIDATES, index.ntotal)
    scores, indices = index.search(query_vec, n_search)

    # Filter by mode: "text" keeps only type=="text"; others keep text+table
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = meta[idx]
        if mode == "text" and item["type"] != "text":
            continue
        results.append({
            "id": item["id"],
            "doc": item["doc"],
            "page": item["page"],
            "type": item["type"],
            "text": item["text"],
            "score": round(float(score), 4),
        })
        if len(results) >= top_k:
            break

    # Label evidence items E1, E2, ...
    for i, r in enumerate(results):
        r["evidence_id"] = f"E{i + 1}"

    return {
        "texts": results,
        "tables": [],
        "images": [],
        "best_text_score": results[0]["score"] if results else 0.0,
        "best_image_score": 0.0,
        "insufficient": len(results) == 0,
    }
