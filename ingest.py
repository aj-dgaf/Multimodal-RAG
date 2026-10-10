"""
ingest.py — Phase 2: Tables, images, CLIP index

Parses every PDF in data/pdfs/ to extract:
1. Text chunks (~200 words).
2. Tables (converted to Markdown).
3. Images (filtered by size, redundancy, and converted to RGB, or rendered vector pages).

Embeds text+tables with bge-small into data/index/text.index.
Embeds images with OpenCLIP into data/index/image.index.

Usage:
    python ingest.py
"""

import json
import hashlib
import re
import pymupdf
import numpy as np
import faiss
import torch
import open_clip
from PIL import Image
from sentence_transformers import SentenceTransformer
from pathlib import Path
import io

import config

# --- Utilities ---

def get_pdf_paths() -> list[Path]:
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: No PDFs found in {config.PDF_DIR}. Add PDFs and re-run.")
    return pdfs

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
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

def find_caption(page, image_rect) -> str:
    """Find a caption near the image."""
    caption_pattern = re.compile(r"^(Figure|Fig\.|Chart|Diagram)\s*\d+", re.IGNORECASE)
    blocks = page.get_text("blocks")
    closest_caption = ""
    min_dist = 999999
    
    for b in blocks:
        text = b[4].strip()
        if not text: continue
        if caption_pattern.search(text.replace("\n", " ")):
            dist = b[1] - image_rect[3]
            if 0 <= dist < min_dist:
                min_dist = dist
                closest_caption = text.replace("\n", " ")
                
    return closest_caption

# --- Extraction ---

def process_pdf(pdf_path: Path) -> tuple[list[dict], list[dict]]:
    """Extract text, tables, and images from a single PDF."""
    doc = pymupdf.open(str(pdf_path))
    doc_name = pdf_path.name
    
    doc_text_items = []
    doc_image_items = []
    
    # Hash -> set of page numbers (for logo deduplication)
    image_hashes = {}
    raw_images = []
    
    for page in doc:
        pnum = page.number + 1
        
        # 1. Tables
        try:
            tables = page.find_tables()
            for tab in tables:
                if hasattr(tab, "to_markdown"):
                    md = tab.to_markdown()
                else:
                    lines = []
                    for row in tab.extract():
                        lines.append(" | ".join([str(c).replace('\n', ' ') if c else "" for c in row]))
                    md = "\n".join(lines)
                if md.strip():
                    doc_text_items.append({
                        "doc": doc_name,
                        "page": pnum,
                        "type": "table",
                        "text": md.strip()
                    })
        except Exception as e:
            print(f"    Warning: table extraction failed on p.{pnum} of {doc_name}: {e}")
                
        # 2. Text
        text = page.get_text().strip()
        if text:
            for chunk in chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                doc_text_items.append({
                    "doc": doc_name,
                    "page": pnum,
                    "type": "text",
                    "text": chunk
                })
                
        # 3. Images
        images_on_page = page.get_images(full=True)
        drawings = page.get_drawings()
        kept_image_on_page = False
        
        for img_info in images_on_page:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                img = Image.open(io.BytesIO(image_bytes))
                
                if img.width < config.MIN_IMAGE_SIZE or img.height < config.MIN_IMAGE_SIZE:
                    continue
                    
                img = img.convert("RGB")
                img_hash = hashlib.md5(img.tobytes()).hexdigest()
                
                if img_hash not in image_hashes:
                    image_hashes[img_hash] = set()
                image_hashes[img_hash].add(pnum)
                
                rects = page.get_image_rects(xref)
                rect = rects[0] if rects else pymupdf.Rect(0,0,0,0)
                caption = find_caption(page, rect)
                
                raw_images.append({
                    "doc": doc_name,
                    "page": pnum,
                    "img_hash": img_hash,
                    "img_obj": img,
                    "caption": caption,
                    "xref": xref
                })
                kept_image_on_page = True
                
            except Exception as e:
                print(f"    Warning: Failed to extract image xref {xref} on {doc_name} p.{pnum}")
                
        # 4. Vector-chart fallback
        if not kept_image_on_page and len(drawings) > config.VECTOR_DRAWING_THRESHOLD:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Use LANCZOS instead of deprecated ANTIALIAS
            img.thumbnail((config.MAX_IMAGE_EDGE, config.MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            image_hashes[img_hash] = {pnum}
            
            raw_images.append({
                "doc": doc_name,
                "page": pnum,
                "img_hash": img_hash,
                "img_obj": img,
                "caption": "",
                "xref": f"vector_{pnum}"
            })
            
    page_count = doc.page_count
    doc.close()
    
    # Process raw images: filter duplicates, downscale, save
    out_dir = config.IMAGE_DIR / doc_name.replace(".pdf", "")
    if raw_images and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        
    for r_img in raw_images:
        # Filter repeat images (logos)
        if len(image_hashes[r_img["img_hash"]]) >= config.LOGO_REPEAT_THRESHOLD:
            continue
            
        img = r_img["img_obj"]
        img.thumbnail((config.MAX_IMAGE_EDGE, config.MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        
        save_path = out_dir / f"p{r_img['page']}_{r_img['xref']}.png"
        img.save(save_path, "PNG")
        
        doc_image_items.append({
            "doc": r_img["doc"],
            "page": r_img["page"],
            "path": str(save_path.relative_to(config.BASE_DIR)).replace("\\", "/"),
            "caption": r_img["caption"]
        })
        
    stats = {
        "pages": page_count,
        "text_chunks": sum(1 for t in doc_text_items if t["type"] == "text"),
        "tables": sum(1 for t in doc_text_items if t["type"] == "table"),
        "images_kept": len(doc_image_items),
        "images_dropped": len(raw_images) - len(doc_image_items),
        "vector_renders": sum(1 for r in raw_images if str(r["xref"]).startswith("vector_"))
    }
        
    return doc_text_items, doc_image_items, stats


def build_corpus(pdf_paths: list[Path]) -> tuple[list[dict], list[dict]]:
    all_texts = []
    all_images = []
    text_id = 0
    image_id = 0
    
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


# --- Embedding & FAISS ---

def embed_texts(chunks: list[dict]):
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not chunks: return

    print(f"\nEmbedding {len(chunks)} text/table chunks with {config.TEXT_EMBED_MODEL}...")
    model = SentenceTransformer(config.TEXT_EMBED_MODEL, device=config.DEVICE)
    texts = [c["text"] for c in chunks]
    
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    faiss.write_index(index, str(config.INDEX_DIR / "text.index"))
    with open(config.INDEX_DIR / "text_meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    print(f"Saved text index: {index.ntotal} vectors")


def embed_images(images: list[dict]):
    if not images:
        print("No images to embed.")
        return
        
    print(f"\nEmbedding {len(images)} images with {config.CLIP_MODEL}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED, device=config.DEVICE
    )
    model.eval()
    
    embeddings = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            preprocessed = []
            for item in batch:
                abs_path = config.BASE_DIR / item["path"]
                img = Image.open(abs_path).convert("RGB")
                preprocessed.append(preprocess(img))
                
            tensor_batch = torch.stack(preprocessed).to(config.DEVICE)
            image_features = model.encode_image(tensor_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy())
            print(f"  Processed {min(i+batch_size, len(images))}/{len(images)} images")
            
    embeddings = np.vstack(embeddings).astype(np.float32)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    faiss.write_index(index, str(config.INDEX_DIR / "image.index"))
    with open(config.INDEX_DIR / "image_meta.json", "w", encoding="utf-8") as f:
        json.dump(images, f, ensure_ascii=False, indent=1)
    print(f"Saved image index: {index.ntotal} vectors")


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    
    print("=== ingest.py: Phase 2 (Text, Tables, Images) ===\n")
    pdf_paths = get_pdf_paths()
    if not pdf_paths: return
    
    print("--- Extracting content ---")
    texts, images = build_corpus(pdf_paths)
    
    print(f"\nTotal text/table chunks: {len(texts)}")
    print(f"Total images kept: {len(images)}")
    
    embed_texts(texts)
    embed_images(images)
    
    print("\n=== Ingestion complete ===")

if __name__ == "__main__":
    main()
