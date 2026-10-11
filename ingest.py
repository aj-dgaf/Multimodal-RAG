"""
ingest.py — PDF ingestion pipeline

Parses every PDF in data/pdfs/ to extract:
  1. Text chunks (~200 words with ~30-word overlap).
  2. Tables (converted to Markdown).
  3. Images (filtered by size and redundancy, converted to RGB).

Embeds text+tables with bge-small into data/index/text.index.
Embeds images with OpenCLIP into data/index/image.index.

Usage:
    python ingest.py
"""

import json
import hashlib
import re
import io
from pathlib import Path

import pymupdf
import numpy as np
import faiss
import torch
import open_clip
from PIL import Image
from sentence_transformers import SentenceTransformer

import config

# --- 1. Utilities ---

def get_pdf_paths() -> list[Path]:
    """Return sorted PDF paths; print error if none found."""
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: No PDFs found in {config.PDF_DIR}. Add PDFs and re-run.")
    return pdfs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size]))
        start += chunk_size - overlap
    return chunks


def find_caption(page, image_rect) -> str:
    """Find the nearest caption (Figure/Chart/Diagram) below an image."""
    pattern = re.compile(r"^(Figure|Fig\.|Chart|Diagram)\s*\d+", re.IGNORECASE)
    closest, min_dist = "", 999999

    for b in page.get_text("blocks"):
        text = b[4].strip()
        if not text:
            continue
        if pattern.search(text.replace("\n", " ")):
            dist = b[1] - image_rect[3]
            if 0 <= dist < min_dist:
                min_dist = dist
                closest = text.replace("\n", " ")
    return closest


# --- 2. Per-page extraction ---

def _extract_tables(page, doc_name: str, pnum: int,
                    doc_text_items: list[dict]) -> None:
    """Extract markdown tables from a PDF page."""
    try:
        tables = page.find_tables()
        for tab in tables:
            if hasattr(tab, "to_markdown"):
                md = tab.to_markdown()
            else:
                lines = []
                for row in tab.extract():
                    cells = [str(c).replace("\n", " ") if c else "" for c in row]
                    lines.append(" | ".join(cells))
                md = "\n".join(lines)
            if md.strip():
                doc_text_items.append({
                    "doc": doc_name, "page": pnum,
                    "type": "table", "text": md.strip(),
                })
    except Exception as e:
        print(f"    Warning: table extraction failed on p.{pnum} of {doc_name}: {e}")


def _extract_text_chunks(page, doc_name: str, pnum: int,
                         doc_text_items: list[dict]) -> None:
    """Extract and chunk text from a PDF page."""
    text = page.get_text().strip()
    if text:
        for chunk in chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            doc_text_items.append({
                "doc": doc_name, "page": pnum,
                "type": "text", "text": chunk,
            })


def _extract_raster_images(doc, page, doc_name: str, pnum: int,
                           image_hashes: dict, raw_images: list[dict]) -> bool:
    """Extract embedded raster images, filtering by size and converting to RGB."""
    kept = False
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            img = Image.open(io.BytesIO(base_image["image"]))

            if img.width < config.MIN_IMAGE_SIZE or img.height < config.MIN_IMAGE_SIZE:
                continue

            img = img.convert("RGB")
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            image_hashes.setdefault(img_hash, set()).add(pnum)

            rects = page.get_image_rects(xref)
            rect = rects[0] if rects else pymupdf.Rect(0, 0, 0, 0)

            raw_images.append({
                "doc": doc_name, "page": pnum, "img_hash": img_hash,
                "img_obj": img, "caption": find_caption(page, rect), "xref": xref,
            })
            kept = True
        except Exception:
            print(f"    Warning: Failed to extract image xref {xref} on {doc_name} p.{pnum}")
    return kept


def _render_vector_page(page, doc_name: str, pnum: int,
                        image_hashes: dict, raw_images: list[dict]) -> None:
    """Render a page with many vector drawings to PNG as a chart fallback."""
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.thumbnail((config.MAX_IMAGE_EDGE, config.MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)

    img_hash = hashlib.md5(img.tobytes()).hexdigest()
    image_hashes.setdefault(img_hash, set()).add(pnum)

    raw_images.append({
        "doc": doc_name, "page": pnum, "img_hash": img_hash,
        "img_obj": img, "caption": "", "xref": f"vector_{pnum}",
    })


# --- 3. Image filtering and saving ---

def _save_filtered_images(doc_name: str, raw_images: list[dict],
                          image_hashes: dict) -> list[dict]:
    """Save deduplicated and downscaled images to disk."""
    doc_image_items = []
    out_dir = config.IMAGE_DIR / doc_name.replace(".pdf", "")
    if raw_images and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    for r_img in raw_images:
        # Skip images that repeat across many pages (logos/headers)
        if len(image_hashes[r_img["img_hash"]]) >= config.LOGO_REPEAT_THRESHOLD:
            continue

        img = r_img["img_obj"]
        img.thumbnail((config.MAX_IMAGE_EDGE, config.MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)

        save_path = out_dir / f"p{r_img['page']}_{r_img['xref']}.png"
        img.save(save_path, "PNG")

        doc_image_items.append({
            "doc": r_img["doc"], "page": r_img["page"],
            "path": str(save_path.relative_to(config.BASE_DIR)).replace("\\", "/"),
            "caption": r_img["caption"],
        })
    return doc_image_items


# --- 4. Document-level processing ---

def process_pdf(pdf_path: Path) -> tuple[list[dict], list[dict], dict]:
    """Extract text, tables, and images from a single PDF."""
    doc = pymupdf.open(str(pdf_path))
    doc_name = pdf_path.name

    doc_text_items = []
    image_hashes = {}
    raw_images = []

    for page in doc:
        pnum = page.number + 1
        _extract_tables(page, doc_name, pnum, doc_text_items)
        _extract_text_chunks(page, doc_name, pnum, doc_text_items)

        kept = _extract_raster_images(doc, page, doc_name, pnum, image_hashes, raw_images)
        # Vector-chart fallback: render pages with many drawings but no kept images
        if not kept and len(page.get_drawings()) > config.VECTOR_DRAWING_THRESHOLD:
            _render_vector_page(page, doc_name, pnum, image_hashes, raw_images)

    page_count = doc.page_count
    doc.close()

    doc_image_items = _save_filtered_images(doc_name, raw_images, image_hashes)

    stats = {
        "pages": page_count,
        "text_chunks": sum(1 for t in doc_text_items if t["type"] == "text"),
        "tables": sum(1 for t in doc_text_items if t["type"] == "table"),
        "images_kept": len(doc_image_items),
        "images_dropped": len(raw_images) - len(doc_image_items),
        "vector_renders": sum(1 for r in raw_images if str(r["xref"]).startswith("vector_")),
    }
    return doc_text_items, doc_image_items, stats


def build_corpus(pdf_paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Process all PDFs and assign global IDs to each chunk and image."""
    all_texts, all_images = [], []
    text_id, image_id = 0, 0

    for pdf in pdf_paths:
        t_items, i_items, stats = process_pdf(pdf)

        for t in t_items:
            t["id"] = f"C{text_id}"
            all_texts.append(t)
            text_id += 1

        for i in i_items:
            i["id"] = f"I{image_id}"
            all_images.append(i)
            image_id += 1

        print(f"  {pdf.name}: pages={stats['pages']}, text={stats['text_chunks']}, "
              f"tables={stats['tables']}, img_kept={stats['images_kept']}, "
              f"img_dropped={stats['images_dropped']}, vectors={stats['vector_renders']}")

    return all_texts, all_images


# --- 5. Embedding & FAISS ---

def embed_texts(chunks: list[dict]) -> None:
    """Embed text/table chunks with bge-small and save to FAISS."""
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not chunks:
        return

    print(f"\nEmbedding {len(chunks)} text/table chunks with {config.TEXT_EMBED_MODEL}...")
    model = SentenceTransformer(config.TEXT_EMBED_MODEL, device=config.DEVICE)
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(config.INDEX_DIR / "text.index"))
    with open(config.INDEX_DIR / "text_meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    print(f"Saved text index: {index.ntotal} vectors")


def _embed_image_batch(model, preprocess, images: list[dict],
                       start: int, batch_size: int) -> np.ndarray:
    """Preprocess and CLIP-encode a batch of images."""
    batch = images[start : start + batch_size]
    preprocessed = []
    for item in batch:
        img = Image.open(config.BASE_DIR / item["path"]).convert("RGB")
        preprocessed.append(preprocess(img))

    tensor = torch.stack(preprocessed).to(config.DEVICE)
    features = model.encode_image(tensor)
    features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


def embed_images(images: list[dict]) -> None:
    """Embed images with CLIP and save to FAISS."""
    if not images:
        print("No images to embed.")
        return

    print(f"\nEmbedding {len(images)} images with {config.CLIP_MODEL}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED, device=config.DEVICE
    )
    model.eval()

    batch_size = 32
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            embeddings.append(_embed_image_batch(model, preprocess, images, i, batch_size))
            print(f"  Processed {min(i + batch_size, len(images))}/{len(images)} images")

    embeddings = np.vstack(embeddings).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(config.INDEX_DIR / "image.index"))
    with open(config.INDEX_DIR / "image_meta.json", "w", encoding="utf-8") as f:
        json.dump(images, f, ensure_ascii=False, indent=1)
    print(f"Saved image index: {index.ntotal} vectors")


# --- 6. CLI ---

def main() -> None:
    """Run the full ingestion pipeline."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== ingest.py: PDF Ingestion Pipeline ===\n")
    pdf_paths = get_pdf_paths()
    if not pdf_paths:
        return

    print("--- Extracting content ---")
    texts, images = build_corpus(pdf_paths)

    print(f"\nTotal text/table chunks: {len(texts)}")
    print(f"Total images kept: {len(images)}")

    embed_texts(texts)
    embed_images(images)

    print("\n=== Ingestion complete ===")


if __name__ == "__main__":
    main()
