"""
ingest.py — Phase 1: Text-only ingestion

Parses every PDF in data/pdfs/ into ~200-word text chunks with ~30-word
overlap, embeds them with bge-small-en-v1.5, and saves a FAISS IndexFlatIP
plus a metadata JSON file to data/index/.

Phase 2 will extend this file to also extract tables and images.

Usage:
    python ingest.py
"""

import json
import pymupdf
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

import config


# --- 1. Discover PDFs ---

def get_pdf_paths() -> list[Path]:
    """Return sorted list of PDF paths in the configured PDF directory."""
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: No PDFs found in {config.PDF_DIR}. Add PDFs and re-run.")
    return pdfs


# --- 2. Extract text from a single PDF ---

def extract_text_pages(pdf_path: Path) -> list[dict]:
    """Return a list of {page, text} dicts for every page with text."""
    doc = pymupdf.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append({"page": page.number + 1, "text": text})
    doc.close()
    return pages


# --- 3. Chunk text into ~200-word pieces with ~30-word overlap ---

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into word-based chunks of chunk_size with overlap."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_text_chunks(pdf_paths: list[Path]) -> list[dict]:
    """Build text chunks from all PDFs. Each chunk: {id, doc, page, type, text}."""
    all_chunks = []
    chunk_id = 0
    for pdf_path in pdf_paths:
        doc_name = pdf_path.name
        pages = extract_text_pages(pdf_path)
        doc_chunks = 0
        for page_info in pages:
            chunks = chunk_text(
                page_info["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP
            )
            for chunk in chunks:
                all_chunks.append({
                    "id": f"C{chunk_id}",
                    "doc": doc_name,
                    "page": page_info["page"],
                    "type": "text",
                    "text": chunk,
                })
                chunk_id += 1
                doc_chunks += 1
        print(f"  {doc_name}: {len(pages)} pages, {doc_chunks} text chunks")
    return all_chunks


# --- 4. Embed and build FAISS index ---

def embed_and_save(chunks: list[dict], model: SentenceTransformer) -> None:
    """Embed all text chunks, L2-normalise, build FAISS IndexFlatIP, and save."""
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    texts = [c["text"] for c in chunks]
    print(f"\nEmbedding {len(texts)} chunks with {config.TEXT_EMBED_MODEL}...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss_path = config.INDEX_DIR / "text.index"
    faiss.write_index(index, str(faiss_path))
    print(f"Saved FAISS index ({index.ntotal} vectors, dim={dim}) to {faiss_path}")

    # Save metadata (everything except the embedding)
    meta_path = config.INDEX_DIR / "text_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    print(f"Saved metadata to {meta_path}")


# --- 5. Main ---

def main() -> None:
    """Run the full text ingestion pipeline."""
    print("=== ingest.py: Text Ingestion ===\n")

    pdf_paths = get_pdf_paths()
    if not pdf_paths:
        return

    print(f"Found {len(pdf_paths)} PDFs in {config.PDF_DIR}\n")

    # Build text chunks
    print("--- Extracting and chunking text ---")
    chunks = build_text_chunks(pdf_paths)
    print(f"\nTotal text chunks: {len(chunks)}")

    if not chunks:
        print("ERROR: No text chunks were extracted. Check your PDFs.")
        return

    # Load embedding model and embed
    print(f"\nLoading text embedding model: {config.TEXT_EMBED_MODEL}")
    model = SentenceTransformer(config.TEXT_EMBED_MODEL, device=config.DEVICE)
    embed_and_save(chunks, model)

    print("\n=== Ingestion complete ===")


if __name__ == "__main__":
    main()
