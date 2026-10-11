"""
app.py — FastAPI backend

Loads models and FAISS indexes once at startup. Serves the single-page
website from static/index.html and exposes the /api/ask endpoint that
calls retriever + generator. Also serves extracted images from data/images/.

Usage:
    uvicorn app:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import retriever
import generator

# --- 1. Startup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models and FAISS indices so the first request is fast."""
    print("Loading models and FAISS indices...")
    try:
        retriever._load_text_components()
        retriever._load_image_components()
    except Exception as e:
        print(f"Warning during startup: {e}")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Serve extracted images as static files
if config.IMAGE_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(config.IMAGE_DIR)), name="images")


# --- 2. Request model ---

class AskRequest(BaseModel):
    """Validated request body for /api/ask."""
    question: str = Field(..., max_length=500, min_length=1)
    mode: str = Field(..., pattern=r"^(text|text\+tables|multimodal)$")


# --- 3. Response helpers ---

def _format_evidence(ev: dict) -> tuple[dict, dict]:
    """Convert internal evidence dicts to the API response format."""
    api_ev = {"texts": [], "tables": [], "images": []}
    ev_dict = {}

    for t in ev.get("texts", []):
        api_ev["texts"].append({
            "id": t["evidence_id"], "doc": t["doc"],
            "page": t["page"], "text": t["text"], "score": t["score"],
        })
        ev_dict[t["evidence_id"]] = {"doc": t["doc"], "page": t["page"], "type": t["type"]}

    for tb in ev.get("tables", []):
        api_ev["tables"].append({
            "id": tb["evidence_id"], "doc": tb["doc"],
            "page": tb["page"], "markdown": tb["text"], "score": tb["score"],
        })
        ev_dict[tb["evidence_id"]] = {"doc": tb["doc"], "page": tb["page"], "type": tb["type"]}

    for img in ev.get("images", []):
        url = "/" + img["path"].replace("data/images/", "images/")
        api_ev["images"].append({
            "id": img["evidence_id"], "doc": img["doc"],
            "page": img["page"], "url": url,
            "caption": img["caption"], "score": img["score"],
        })
        ev_dict[img["evidence_id"]] = {"doc": img["doc"], "page": img["page"], "type": "image"}

    return api_ev, ev_dict


def _format_citations(raw_citations: list[str], ev_dict: dict) -> list[dict]:
    """Map raw citation IDs to structured citation objects."""
    return [
        {"id": cid, "doc": ev_dict[cid]["doc"],
         "page": ev_dict[cid]["page"], "type": ev_dict[cid]["type"]}
        for cid in raw_citations if cid in ev_dict
    ]


# --- 4. Routes ---

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page website."""
    index_path = config.BASE_DIR / "static" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>static/index.html not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/docs")
async def get_docs():
    """Return the list of PDFs in the corpus."""
    pdfs = [p.name for p in config.PDF_DIR.glob("*.pdf")]
    return {"docs": sorted(pdfs)}


@app.post("/api/ask")
async def ask(req: AskRequest):
    """Answer a question using the specified retrieval mode."""
    try:
        res = generator.generate(req.question, mode=req.mode)
        api_evidence, ev_dict = _format_evidence(res.get("evidence", {}))

        return {
            "mode": req.mode,
            "answer": res.get("answer", ""),
            "answer_found": res.get("answer_found", False),
            "citations": _format_citations(res.get("citations", []), ev_dict),
            "evidence": api_evidence,
            "latency_ms": res.get("latency_ms", 0),
        }
    except Exception as e:
        print(f"Error handling /api/ask: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the answer.",
        )
