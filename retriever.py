"""
retriever.py — Multimodal retrieval

Loads FAISS indices (text and image) and metadata built by ingest.py.
Embeds the query for text (bge-small) and images (CLIP text encoder).
Retrieves top texts/tables and top images, plus page-linked images.

Usage:
    Imported by generator.py and app.py
"""

import json
import numpy as np
import faiss
import torch
import open_clip
from sentence_transformers import SentenceTransformer

import config

# --- 1. Constants ---

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- 2. Lazy-loaded singletons ---

_text_index = None
_text_meta = None
_text_model = None

_image_index = None
_image_meta = None
_clip_model = None
_clip_tokenizer = None


# --- 3. Index loading ---

def _load_text_components() -> tuple:
    """Load the text FAISS index, metadata, and embedding model."""
    global _text_index, _text_meta, _text_model
    if _text_index is not None:
        return _text_index, _text_meta, _text_model

    index_path = config.INDEX_DIR / "text.index"
    meta_path = config.INDEX_DIR / "text_meta.json"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Text index not found. Run ingest.py first.")

    _text_index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        _text_meta = json.load(f)

    _text_model = SentenceTransformer(config.TEXT_EMBED_MODEL, device=config.DEVICE)
    print(f"Loaded text index: {_text_index.ntotal} vectors")
    return _text_index, _text_meta, _text_model


def _load_image_components() -> tuple:
    """Load the image FAISS index, metadata, CLIP model and tokenizer."""
    global _image_index, _image_meta, _clip_model, _clip_tokenizer
    if _image_index is not None:
        return _image_index, _image_meta, _clip_model, _clip_tokenizer

    index_path = config.INDEX_DIR / "image.index"
    meta_path = config.INDEX_DIR / "image_meta.json"
    if not index_path.exists() or not meta_path.exists():
        return None, None, None, None

    _image_index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        _image_meta = json.load(f)

    _clip_model, _, _ = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED, device=config.DEVICE
    )
    _clip_model.eval()
    _clip_tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)

    print(f"Loaded image index: {_image_index.ntotal} vectors")
    return _image_index, _image_meta, _clip_model, _clip_tokenizer


# --- 4. Text search ---

def _search_text_index(question: str, mode: str) -> tuple[list[dict], float]:
    """Embed the question with bge-small and search the text/table FAISS index."""
    text_idx, text_meta, text_model = _load_text_components()

    query_vec = text_model.encode(
        [BGE_QUERY_PREFIX + question], normalize_embeddings=True
    )
    query_vec = np.array(query_vec, dtype=np.float32)

    n_search = min(config.TEXT_CANDIDATES, text_idx.ntotal)
    t_scores, t_indices = text_idx.search(query_vec, n_search)

    results = []
    for score, idx in zip(t_scores[0], t_indices[0]):
        if idx < 0:
            continue
        item = text_meta[idx]
        # "text" mode excludes tables; other modes keep both
        if mode == "text" and item["type"] != "text":
            continue
        results.append({
            "doc": item["doc"], "page": item["page"],
            "type": item["type"], "text": item["text"],
            "score": round(float(score), 4),
        })
        if len(results) >= config.TOP_K_TEXT:
            break

    best_score = results[0]["score"] if results else 0.0
    return results, best_score


# --- 5. Image search ---

def _search_image_index(question: str, text_results: list[dict]) -> tuple[list[dict], float]:
    """CLIP text-encode the question, search the image index, and add page-linked images."""
    img_idx, img_meta, clip_model, clip_tok = _load_image_components()
    if img_idx is None or img_idx.ntotal == 0:
        return [], 0.0

    # CLIP text encode (tokenizer handles truncation at 77 tokens)
    text_tokens = clip_tok([question])
    if text_tokens.shape[1] > 77:
        text_tokens = text_tokens[:, :77]

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens.to(config.DEVICE))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_vec = text_features.cpu().numpy().astype(np.float32)

    n_img = min(config.TOP_K_IMAGES, img_idx.ntotal)
    i_scores, i_indices = img_idx.search(clip_vec, n_img)

    results = []
    for score, idx in zip(i_scores[0], i_indices[0]):
        if idx < 0:
            continue
        item = img_meta[idx]
        results.append({
            "doc": item["doc"], "page": item["page"], "type": "image",
            "path": item["path"], "caption": item["caption"],
            "score": round(float(score), 4),
        })

    best_score = results[0]["score"] if results else 0.0

    # Page-linked images: images on the same doc+page as the top-3 text hits
    top3_pages = {(r["doc"], r["page"]) for r in text_results[:3]}
    existing_paths = {r["path"] for r in results}
    for img in img_meta:
        if (img["doc"], img["page"]) in top3_pages and img["path"] not in existing_paths:
            results.append({
                "doc": img["doc"], "page": img["page"], "type": "image",
                "path": img["path"], "caption": img["caption"],
                "score": 0.0,
            })
            existing_paths.add(img["path"])

    return results[:config.MAX_IMAGES], best_score


# --- 6. Public API ---

def search(question: str, mode: str = "text") -> dict:
    """Retrieve evidence for a question, filtered by mode."""
    text_results, best_text_score = _search_text_index(question, mode)

    image_results, best_image_score = [], 0.0
    if mode == "multimodal":
        image_results, best_image_score = _search_image_index(question, text_results)

    # Evidence gate: skip the LLM call if all scores are too low
    insufficient = (
        (best_text_score < config.TEXT_SCORE_THRESHOLD
         and best_image_score < config.IMAGE_SCORE_THRESHOLD)
        or (not text_results and not image_results)
    )

    # Label every evidence item E1, E2, ...
    counter = 1
    for r in text_results + image_results:
        r["evidence_id"] = f"E{counter}"
        counter += 1

    return {
        "texts": [r for r in text_results if r["type"] == "text"],
        "tables": [r for r in text_results if r["type"] == "table"],
        "images": image_results,
        "best_text_score": best_text_score,
        "best_image_score": best_image_score,
        "insufficient": insufficient,
    }
