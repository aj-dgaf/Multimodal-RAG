"""
app.py — Phase 5: FastAPI backend
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import os

import config
import retriever
import generator

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models and FAISS indices...")
    try:
        retriever._load_text_components()
        retriever._load_image_components()
    except Exception as e:
        print(f"Warning during startup: {e}")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Mount static images
if config.IMAGE_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(config.IMAGE_DIR)), name="images")

class AskRequest(BaseModel):
    question: str = Field(..., max_length=500, min_length=1)
    mode: str = Field(..., pattern="^(text|text\\+tables|multimodal)$")

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = config.BASE_DIR / "static" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>static/index.html not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/docs")
async def get_docs():
    pdfs = [p.name for p in config.PDF_DIR.glob("*.pdf")]
    return {"docs": sorted(pdfs)}

@app.post("/api/ask")
async def ask(req: AskRequest):
    try:
        res = generator.generate(req.question, mode=req.mode)
        
        # Build API compliant evidence dict
        api_evidence = {"texts": [], "tables": [], "images": []}
        ev = res.get("evidence", {})
        
        ev_dict = {}
        
        for t in ev.get("texts", []):
            item = {
                "id": t["evidence_id"],
                "doc": t["doc"],
                "page": t["page"],
                "text": t["text"],
                "score": t["score"]
            }
            api_evidence["texts"].append(item)
            ev_dict[t["evidence_id"]] = {"doc": t["doc"], "page": t["page"], "type": t["type"]}
            
        for tb in ev.get("tables", []):
            item = {
                "id": tb["evidence_id"],
                "doc": tb["doc"],
                "page": tb["page"],
                "markdown": tb["text"], # spec uses 'markdown'
                "score": tb["score"]
            }
            api_evidence["tables"].append(item)
            ev_dict[tb["evidence_id"]] = {"doc": tb["doc"], "page": tb["page"], "type": tb["type"]}
            
        for img in ev.get("images", []):
            # The path is relative to BASE_DIR (e.g. data/images/doc/p1_1.png)
            # The URL should be /images/doc/p1_1.png
            url = "/" + img["path"].replace("data/images/", "images/")
            item = {
                "id": img["evidence_id"],
                "doc": img["doc"],
                "page": img["page"],
                "url": url,
                "caption": img["caption"],
                "score": img["score"]
            }
            api_evidence["images"].append(item)
            ev_dict[img["evidence_id"]] = {"doc": img["doc"], "page": img["page"], "type": "image"}
            
        citations = []
        for cid in res.get("citations", []):
            if cid in ev_dict:
                citations.append({
                    "id": cid,
                    "doc": ev_dict[cid]["doc"],
                    "page": ev_dict[cid]["page"],
                    "type": ev_dict[cid]["type"]
                })
                
        return {
            "mode": req.mode,
            "answer": res.get("answer", ""),
            "answer_found": res.get("answer_found", False),
            "citations": citations,
            "evidence": api_evidence,
            "latency_ms": res.get("latency_ms", 0)
        }
        
    except Exception as e:
        print(f"Error handling /api/ask: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the answer.")
