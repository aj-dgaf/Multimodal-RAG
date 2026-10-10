# ANTIGRAVITY_BRIEF.md — Multimodal RAG Mini Project

> **Read this whole file before doing anything.** It is the single source of truth for the project. Follow it exactly. If something is unclear or conflicts with this file, **stop and ask me**; do not guess.

---

## 0. Working Agreement (how you must work)

1. Work **one phase at a time** (Section 11). At the end of each phase: run the acceptance checks, show me the **real output**, list every file you created or changed, run `git status`, commit, then **STOP and wait for my approval** before starting the next phase.
2. This is a **college mini project**. I must read, demo and explain every line in a viva. **Simplicity beats cleverness.** No frameworks, abstractions or "best-practice scaffolding" beyond what this file specifies.
3. **Never create a file or folder that is not in the Allowed File Map (Section 4).** No `utils/`, `helpers/`, `tests/`, notebooks, scratch files, notes, plan files or extra docs. If you think another file is needed, ask first.
4. I am on **Windows (PowerShell)**. Give Windows-compatible commands. Use `pathlib` for all paths, open text files with `encoding="utf-8"`, and avoid Linux-only tools (`grep`, `rm`, `export`).
5. Never print, log or commit secrets. The Gemini API key lives only in a local `.env` that I create.
6. Do not add dependencies beyond Section 6 without asking.
7. Do not refactor files outside the current phase unless I ask.

**Starting state:** I have created an **empty project folder** (this is the workspace you are opened in). The only file in it may be this brief. **That folder IS the project root.** Do not create another nested project folder inside it, and do not rename it. Build everything from scratch inside it, starting with Phase 0.

**Things I will do myself (do not try to do them):** create the Gemini API key and `.env`, download the 12 PDFs into `data/pdfs/` (after Phase 0 creates that folder), and write the evaluation questions and gold answers.

**Things you must ask me for:** the GitHub repo name and my GitHub username (before adding a git remote, Phase 0). The repo is **public**.

---

## 1. Project Summary

**Title:** Multimodal RAG: Retrieval-Augmented Generation over Text, Tables and Images
**Student:** Anusha Jain (C219) · Natural Language Processing mini project

**Problem:** Real documents (annual reports, product manuals, textbooks) mix text, tables and images. Standard text-only RAG embeds only text, so evidence that lives in a table or a chart/diagram is lost.

**Solution:** A RAG system that parses PDFs into text chunks, tables (as markdown) and images; embeds text/tables with a sentence-transformer and images with CLIP; retrieves across all three; and gives the evidence (text **and** images) to a vision-capable LLM (Gemini) that returns a **grounded answer with source document and page citations**, or says the answer is not in the documents.

**Core technical idea (must be visible in the code and README):** CLIP embeds text and images into the **same vector space**, so a text query can be matched directly to images without captioning.

**Intended users:** analysts, support engineers, students querying long technical PDFs.

**Final deliverable:** a clean Python project plus a **simple website** (FastAPI + one HTML page), pushed to a public GitHub repo.

---

## 2. Confirmed Decisions

| Topic | Decision |
|---|---|
| Corpus | 12 PDFs, India-focused mix: 4 annual reports, 4 product manuals, 4 NCERT textbook chapters |
| Generation LLM | **Gemini free tier** via the `google-genai` SDK (vision-capable Flash model; check the current model name and free-tier limits in the official docs and set it once in `config.py`) |
| Backend | **FastAPI + uvicorn** |
| Frontend | **One** `static/index.html` (inline CSS + vanilla JS, no framework, no build step) |
| Orchestration | **Plain Python**. **Do NOT use LlamaIndex or LangChain** (simpler to explain) |
| Vector store | FAISS (`faiss-cpu`), `IndexFlatIP` on L2-normalised vectors (= cosine) |
| Text embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers |
| Image embeddings | `open_clip` `ViT-B-32`, pretrained `laion2b_s34b_b79k` |
| Evaluation | Own deterministic metrics + Gemini-as-judge + **RAGAS** (time-boxed; see Phase 4) |
| Python | 3.11 |
| OS / hardware | Windows 11, Intel i5, NVIDIA GTX 1650 (4 GB VRAM) |
| Repo | Public GitHub; **PDFs are NOT committed** (copyrighted) |
| Theme | Light; palette and fonts in Section 9 |

---

## 3. Scope

**In scope:** English PDFs with selectable text, tables and images; text chunking; table extraction to markdown; image extraction + CLIP embedding; dense retrieval with FAISS; grounded, cited generation with Gemini (text + images); evaluation of text-only vs text+tables vs full multimodal; a simple website.

**Out of scope (do not build):** audio/video, live ingestion, scanned PDFs / OCR, non-PDF formats, fine-tuning any model, hybrid dense+sparse retrieval, local vision LLMs, authentication, cloud deployment, databases.

---

## 4. Allowed File Map (the ENTIRE repository)

The tree below is relative to the **project root = the current workspace folder** (not a subfolder). Phase 0 creates this entire structure up front, so the layout is visible from day one; later phases only fill the files in.

```text
<project root = current workspace folder>/
├── README.md              # what it is, setup, run, corpus table, results, failure cases, screenshots
├── ANTIGRAVITY_BRIEF.md   # this file
├── requirements.txt       # pinned dependencies
├── .env.example           # GEMINI_API_KEY=your_key_here
├── .gitignore
├── config.py              # EVERY setting in one place
├── ingest.py              # PDFs -> chunks/tables/images -> embeddings -> FAISS indexes
├── retriever.py           # load indexes; search; modes "text", "text+tables", "multimodal"
├── generator.py           # build prompt, call Gemini, parse JSON, validate citations, cache
├── evaluate.py            # run eval set across modes, compute metrics, save results
├── app.py                 # FastAPI server (API + serves the page + serves images)
├── static/
│   └── index.html         # the whole website
└── data/
    ├── pdfs/              # 12 source PDFs (GITIGNORED)
    ├── eval_questions.json
    ├── index/             # GENERATED, gitignored: faiss files + metadata json
    ├── images/            # GENERATED, gitignored: extracted figures
    ├── cache.json         # GENERATED, gitignored: Gemini answer cache
    └── results.json       # GENERATED by evaluate.py, committed (+ results.png)
```

That is **7 code files + 1 HTML page**. Nothing else. Generated charts (`data/results.png`) are the only extra output allowed.

---

## 5. Code Standards (apply to every file)

- Plain Python 3.11. **Functions, not classes** (use a class only if clearly simpler).
- No abstract base classes, decorators, design patterns, plugin systems, dependency injection.
- Every `.py` file starts with a **docstring**: what it does, where it fits in the pipeline, how to run it.
- Every function: one-line docstring + type hints. Keep functions under ~30 lines.
- Comments explain **why**, not what. Number sections inside files (`# --- 1. Load PDFs ---`).
- **All** settings (paths, model names, k values, thresholds, chunk sizes, device) live in `config.py` only. No magic numbers elsewhere.
- No dead code, no commented-out code, no unused imports, no speculative features.
- Minimal logging: a few clear `print` lines (progress and stats). No logging framework.
- CLI entry points use `argparse` only where a file is run directly (`ingest.py`, `generator.py`, `evaluate.py`).
- Handle errors where they realistically happen (missing key, missing index, API failure, bad input) with short, friendly messages. Do not wrap everything in try/except.

---

## 6. Tech Stack and `requirements.txt`

Packages: `pymupdf`, `sentence-transformers`, `open_clip_torch`, `faiss-cpu`, `google-genai`, `fastapi`, `uvicorn`, `python-dotenv`, `pillow`, `numpy`, `matplotlib`, `ragas`.
Pin versions after the first successful install (`pip freeze` only for these packages and their needed pins; keep the file short).

**Windows setup (PowerShell):**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If `faiss-cpu` or `open_clip_torch` fails, report the exact error to me; do not swap libraries.

**Device:** `config.py` sets `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`. The code must run on CPU. The default Windows torch build may be CPU-only; that is acceptable (indexing ~300-500 pages takes minutes). If CUDA out-of-memory occurs, fall back to CPU via config.

**Gemini SDK usage (verify against current official docs before coding):**
```python
from google import genai
from google.genai import types
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
resp = client.models.generate_content(
    model=config.GEMINI_MODEL,
    contents=[text_part_or_str, types.Part.from_bytes(data=img_bytes, mime_type="image/png"), ...],
    config=types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=ANSWER_SCHEMA,   # answer, answer_found, citations
    ),
)
```
Do not use the deprecated `google-generativeai` package.

---

## 7. Architecture

```text
PDFs ──► ingest.py ──► text chunks ─┐
                       tables (md) ─┼─► bge-small ──► FAISS text index  (type: text | table)
                       images ──────┴─► CLIP image ─► FAISS image index
                                                           │
question ──► retriever.py ─► top-k text/tables  +  top-k images (CLIP text->image)
                              + images on the same doc+page as the top text/table hits
                              + evidence gate ("insufficient evidence" -> no LLM call)
                                      │
                                      ▼
                  generator.py ─► Gemini (labelled text + images, JSON out, temp 0)
                                      │  {answer, answer_found, citations[ids]}
                                      ▼
                  app.py (FastAPI) ─► static/index.html
```

### Three retrieval modes (the project's experiment)

| Mode string | Evidence used | Represents |
|---|---|---|
| `"text"` | Plain text chunks only (`page.get_text()`, tables flattened as text) | Standard text-only RAG (Baseline A) |
| `"text+tables"` | Text + structured markdown tables | Ablation (B) |
| `"multimodal"` | Text + tables + images, and images are sent to Gemini | The proposed system (C) |

Implement modes as a **single `mode` argument** in `retriever.py` and `generator.py`. **No duplicated code paths, no separate files per mode.**

---

## 8. Data and Corpus

12 PDFs in `data/pdfs/` (I download them myself; they are **gitignored**):

| Type | Count | Notes |
|---|---|---|
| Annual reports | 4 | Indian companies (e.g. Infosys, a bank, an FMCG, an auto maker); trimmed to ~40-50 pages of tables and charts |
| Product manuals | 4 | e.g. car owner's manual, two-wheeler, washing machine/AC, laptop/phone/printer guide; from manufacturer sites |
| NCERT chapters | 4 | Class 12 Biology, Physics, Chemistry, plus an Economics or Geography chapter |

Constraints: text must be **selectable** (no scanned PDFs); ~20-60 pages each, ~300-500 pages total; English only.

The README must contain a corpus table (title, publisher, URL, pages used) so others can re-download. Do not commit PDFs.

---

## 9. Website Design Spec

### Palette (CSS variables at the top of `index.html`)

| Name | HEX | Role | Variable |
|---|---|---|---|
| Honeydew | `#F6FFE9` | Page background | `--bg` |
| White | `#FFFFFF` | Cards/panels (with Periwinkle border) | `--surface` |
| Periwinkle | `#CAC5E5` | Borders, dividers, inactive tabs, skeleton loader, soft shadow tint | `--border` |
| Dark Ultramarine | `#290087` | All text, headings, header bar, primary buttons, active tab, links, focus ring (text on it: Honeydew) | `--ink`, `--primary` |
| Amethyst | `#A230A4` | Citation pills, Compare-view highlights, hover accents (short labels/icons only) | `--accent` |
| Vanilla Custard | `#F2E0A4` | Example chips, "not found" notice, highlighted evidence | `--highlight` |

Rules: body text is always Dark Ultramarine on Honeydew/White. Amethyst is never used for long paragraphs. Periwinkle is never used for text. (Note: Vanilla Custard is `#F2E0A4`, consistent with its RGB 242, 224, 164.)

### Fonts (do NOT commit font files; do NOT hot-link demo-font sites)

```css
:root {
  --font-heading: "SS Default Lingo", "Astrid Grotesk", "Inter", system-ui, sans-serif;
  --font-ui:      "Astrid Grotesk", "Inter", system-ui, sans-serif;
}
```
- **SS Default Lingo** (Typocracy Studio): display typeface, **headings and project title only** (~20 px and up).
- **Astrid Grotesk** (Schizotype / Eclectotype): all UI, message, body, buttons, citations, evidence text. Use tabular figures for numbers/citations (`font-variant-numeric: tabular-nums`).
- Fonts are picked up from the viewer's installed system fonts. Load **Inter** from Google Fonts as the fallback so the page looks good without them.
- README credit line: "Fonts: SS Default Lingo (Typocracy Studio), Astrid Grotesk (Schizotype / Eclectotype)."

### Look and feel
Light "research tool" feel, centered column (max ~960 px), 12 px rounded cards, soft Periwinkle-tinted shadows, generous whitespace, responsive for laptop and phone. Visible keyboard focus. No external JS libraries.

### Layout and components
1. **Header:** project title (heading font) + one-line description.
2. **Corpus chips row** (document names from `/api/docs`).
3. **Ask bar** (single input + button; Enter submits) with **4-5 example question chips** (Vanilla Custard). Chips are known-working questions I will supply later.
4. **Mode toggle:** `Text-only | Text + Tables | Multimodal | Compare`. **Compare** sends two requests (text-only and multimodal) and shows the two answers side by side. This is the key demo moment.
5. **Answer card:** answer text; **citation pills** like `Annual_Report · p.14` (Amethyst; click scrolls/highlights the matching evidence); confidence-free, plain language. If `answer_found` is false show a clear Vanilla Custard **"Not found in the documents"** notice.
6. **Evidence panel with tabs:** `Text` · `Tables` (markdown rendered as a real HTML table via a tiny JS converter of ~15 lines) · `Images` (thumbnails with doc/page, click for a simple lightbox).
7. **States:** loading skeleton, error message (friendly, no stack traces), empty state, "not found" state.
8. **Footer:** model name and latency.

---

## 10. Backend API Spec (`app.py`)

Load models and indexes **once at startup**. Endpoints:

- `GET /` → serves `static/index.html`
- `GET /api/docs` → `{"docs": ["name1.pdf", ...]}`
- `POST /api/ask` body `{"question": str, "mode": "text" | "text+tables" | "multimodal"}` →
```json
{
  "mode": "multimodal",
  "answer": "string",
  "answer_found": true,
  "citations": [{"id": "E3", "doc": "Infosys_AR_2025-26.pdf", "page": 14, "type": "table"}],
  "evidence": {
    "texts":  [{"id": "E1", "doc": "...", "page": 12, "text": "...", "score": 0.71}],
    "tables": [{"id": "E3", "doc": "...", "page": 14, "markdown": "| a | b |...", "score": 0.69}],
    "images": [{"id": "E7", "doc": "...", "page": 15, "url": "/images/<doc>/p15_1.png", "caption": "Figure 3 ..."}]
  },
  "latency_ms": 1840
}
```
- `/images/...` → static files from `data/images/`.
- Validate input: non-empty, max ~500 characters, mode in the allowed set. Return clear JSON errors (4xx/5xx), never stack traces. Do not enable CORS (same origin).

---

## 11. Phases

Timeline (6 weeks): P0 = week 1, P1 = week 2, P2 = week 3, P3 = week 4, P4 = week 5, P5 + P6 = week 6. Phase 5 only needs Phase 3, so it can be done before Phase 4 if time is short.

### Phase 0: Setup (create everything from scratch)
**Goal:** the complete folder structure, working environment, proof that Gemini accepts text + an image, and a first commit.

**Step 1: Create the structure** (inside the current empty workspace folder; check first that no nested folder is created):
```powershell
mkdir static, data, data\pdfs
```
Create these files (stubs are fine now; each later phase fills them in):

| File | Initial content |
|---|---|
| `config.py` | Real content (see Step 2) |
| `requirements.txt` | Packages from Section 6 (pin after install) |
| `.env.example` | `GEMINI_API_KEY=your_key_here` |
| `.gitignore` | `.env`, `.venv/`, `__pycache__/`, `*.pyc`, `.DS_Store`, `data/pdfs/`, `data/index/`, `data/images/`, `data/cache.json` |
| `README.md` | Title, one-paragraph pitch, "work in progress" note |
| `ingest.py`, `retriever.py`, `generator.py`, `evaluate.py`, `app.py` | **Only** a module docstring (what the file will do, where it fits, how to run it). No other code yet |
| `static/index.html` | A minimal valid placeholder page (title + "coming soon"), with the CSS variables from Section 9 |
| `data/eval_questions.json` | `[]` |

`data/pdfs/` stays empty for now (I will add the PDFs). Do **not** create `data/index/`, `data/images/`, `data/cache.json` or `data/results.json`; the scripts create them when run (use `mkdir(parents=True, exist_ok=True)`). `ingest.py` (Phase 1) must print a clear message if `data/pdfs/` has no PDFs.

**Step 2: `config.py`** with: base paths (pathlib, relative to the file), `DEVICE` (auto-detect CUDA, fall back to CPU), `TEXT_EMBED_MODEL`, `CLIP_MODEL`, `CLIP_PRETRAINED`, `GEMINI_MODEL`, chunk size (~200 words) and overlap (~30), `TOP_K_TEXT` (5), `TOP_K_IMAGES` (3), `MAX_IMAGES` (4), `TEXT_CANDIDATES` (40), min image size (150 px), max image edge (1024 px), logo-repeat threshold (3 pages), vector-drawing threshold, score thresholds (placeholders), cache path, retry settings.

**Step 3: Environment:** create the venv, install requirements (Windows commands in Section 6), then verify in the terminal (**do not save a script**) that PyMuPDF, sentence-transformers, open_clip, faiss and google-genai import, and that Gemini answers a prompt containing one text and one small image. I will have created `.env` with my key; if it is missing, tell me and wait.

**Step 4: Git:** **ask me** for the GitHub repo name and my GitHub username, then `git init`, `git branch -M main`, add the remote, and make the first commit and push.

**Done when:** the structure above exists (show me a tree listing); all imports succeed; Gemini describes the image; `git status` is clean; the first commit is pushed.
**Commit:** `chore: project skeleton`

### Phase 1: Corpus check and text-only RAG baseline
**Precondition:** I have added the PDFs to `data/pdfs/` (created in Phase 0). If the folder is empty, stop and tell me. Confirm every PDF has extractable text.
**Tasks**
- `ingest.py` (text part): for each PDF, `page.get_text()` per page, split into ~200-word chunks with ~30-word overlap; store `{id, doc, page, type: "text", text}`. Embed with bge-small (**normalised**; documents embedded as-is, optionally prefixed with `doc | p.N:`), build `faiss.IndexFlatIP`, save index + metadata JSON to `data/index/`. Print chunks per document.
- `retriever.py` (text part): load index; `search` embeds the query with bge's retrieval instruction prefix (`"Represent this sentence for searching relevant passages: "`) and returns top-k chunks with scores.
- `generator.py` (text part): build a prompt from the chunks, ask Gemini (temperature 0) to answer **only** from the evidence and cite chunk IDs; CLI `python generator.py "question"` prints the answer with doc + page.
**Done when:** `python ingest.py` builds the index; `python generator.py "<question>"` answers 5 of 5 text questions I choose correctly, each with a doc + page citation.
**Commit:** `feat: text-only RAG baseline`

### Phase 2: Tables, images, CLIP index (extend `ingest.py` only)
**Tables**
- `page.find_tables()`; convert each to **markdown** (`table.to_markdown()` if available, else build from `table.extract()`). Store as `{type: "table", doc, page, text: <markdown>}` in the **same text index** (same bge embedder). The `type` tag lets `"text"` mode exclude them.
**Images / figures**
- Extract embedded raster images (`page.get_images()` / `Pixmap`). **Convert CMYK/alpha properly to RGB.**
- **Filter junk:** drop images smaller than 150×150 px; drop images whose hash repeats on 3+ pages (logos/headers).
- **Vector-chart fallback:** if a page has many vector drawings (`page.get_drawings()` above the config threshold) and no kept image, render the page (or its main drawing region) to PNG and treat it as a figure.
- Downscale so the long edge ≤ 1024 px; save to `data/images/<doc_stem>/p<page>_<n>.png`.
- Optional `caption`: nearest text line on the page matching `^(Figure|Fig\.|Chart|Diagram)\s*\d+`.
- Embed images with CLIP (**normalised**) and save a **second** FAISS index + metadata `{id, doc, page, path, caption}`.
**Output:** a per-document summary (pages, text chunks, tables, images kept, images dropped, vector-page renders).
**Done when (manual spot-check required):** stats printed; no document shows 0 tables **and** 0 images unless that is genuinely true; I open 10 random extracted images and 5 random table chunks and confirm they are real content (not logos, not garbled); `python ingest.py` rebuilds everything from scratch in a few minutes.
**Commit:** `feat: table and image parsing, CLIP image index`

### Phase 3: Multimodal retrieval and grounded generation (`retriever.py`, `generator.py`)
**`retrieve(question, mode)`**
1. bge-embed the question; search the text index for `TEXT_CANDIDATES`; **filter by mode** (`"text"` keeps `type == "text"`; the other modes keep `text` + `table`); keep top `TOP_K_TEXT`.
2. **`"multimodal"` only, images:** (a) CLIP **text-encode** the question (**CLIP truncates at 77 tokens**; use the open_clip tokenizer's truncation and keep the query short) and take the top `TOP_K_IMAGES` images; (b) add **page-linked images**: images on the same `doc + page` as the top-3 text/table hits; (c) de-duplicate and cap at `MAX_IMAGES`.
3. **Evidence gate:** if the best text score and best image score are both below thresholds in `config.py`, return an "insufficient evidence" result **without calling Gemini**.
4. Label every evidence item `E1, E2, ...` with doc, page, type.

**`generate(question, mode)`**
- One Gemini request: instructions + labelled text/table evidence + labelled images (each image preceded by a text part such as `Image E7 (doc X, page 15):`).
- Core instruction: *"Answer only from the evidence. If the evidence does not contain the answer, set answer_found to false. Cite evidence IDs. Never invent numbers; quote figures exactly as shown. Refer to a table or figure by its page."*
- Structured JSON via `response_schema`: `{answer: str, answer_found: bool, citations: [evidence id strings]}`; `temperature=0`.
- **Validate citations:** drop IDs not in the provided evidence. If `answer_found` is true but no valid citation remains, retry once, otherwise return the answer flagged as unverified.
- **Robustness:** retry with exponential backoff on HTTP 429/5xx (settings in `config.py`); on repeated failure return the retrieved evidence plus a friendly message (no crash).
- **Cache:** `data/cache.json` keyed by `(mode, normalised question)`; identical requests are instant and use no quota.
- CLI: `python generator.py "question" --mode multimodal` prints the answer, citations and evidence summary.

**Done when:** a table + chart question is answered correctly with the right page; a diagram question is answered correctly in `multimodal`; an **unanswerable** question returns "not found in the documents" (not a guess); a repeated question is served from the cache.
**Commit:** `feat: multimodal retrieval and grounded generation`

### Phase 4: Evaluation (`evaluate.py`, `data/eval_questions.json`)
**I write the questions and gold answers** (you only format the JSON and build the script). Target 35 questions: 10 text, 10 table, 10 figure/diagram, 5 unanswerable. Schema per item: `{id, question, type, gold_answer, gold_doc, gold_page}` (`type` ∈ `text|table|figure|unanswerable`; unanswerable items have null gold fields).

**For each question and each mode** (`text`, `text+tables`, `multimodal`) compute:
1. **Page hit@k:** the gold doc+page appears among retrieved evidence (deterministic, no LLM).
2. **Answer correctness:** Gemini-as-judge compares answer vs gold → `CORRECT | PARTIAL | WRONG` (judge prompt in the file, temperature 0).
3. **Abstention accuracy:** unanswerable → `answer_found == false`; also count false abstentions on answerable questions.
4. **RAGAS** faithfulness and answer relevancy with Gemini as judge. **Time-box this:** wrap in try/except so the script still completes if RAGAS fails; report that it was attempted and why it was dropped if so.
5. Latency per query.

**Outputs:** `data/results.json`, a printed markdown summary table (accuracy by question type × mode, hit@k, abstention, latency), and one bar chart `data/results.png` (matplotlib). Reuse the answer cache and sleep between API calls (free-tier limits). The script must be resumable (cached answers are not re-requested).
**Also:** use the run to tune the evidence-gate thresholds in `config.py`.
**Done when:** `python evaluate.py` produces the table and chart.
**Commit:** `feat: evaluation of text-only vs multimodal`

### Phase 5: Website (`app.py`, `static/index.html` only)
Implement Sections 9 and 10 exactly. Run with `uvicorn app:app --reload` (Windows-friendly).
**Done when:** the page loads; the example questions work in every mode; Compare shows two answers side by side; an unanswerable question shows the "Not found in the documents" state; evidence tabs render text, real HTML tables and clickable image thumbnails; layout works on a phone-width window; the browser console has no errors; fonts fall back gracefully when Astrid Grotesk / SS Default Lingo are not installed.
**Commit:** `feat: FastAPI backend and single-page website`

### Phase 6: Hardening, docs, final push
- **Clean-clone test:** clone the repo into a new folder, create a fresh venv, follow the README exactly (re-download PDFs from the README corpus table, build the index, run the app). Fix whatever breaks.
- **Code review pass** (readability only): remove dead code, add missing docstrings, split functions >30 lines, move magic numbers into `config.py`. **No new features.**
- **README.md:** pitch, architecture diagram (the ASCII one above), setup (Windows commands), run steps, corpus table with source URLs and pages used, results table + chart, **3 concrete failure cases** (question, what was retrieved, why it failed, what would fix it), limitations and future work, website screenshots, font credits.
- Confirm no secrets or PDFs are tracked (`git ls-files`).
- Tag `v1.0` and push (`git push --tags`).
**Commit:** `docs: README, results, final polish`

---

## 12. Reliability Requirements (must be built in, not optional)

Zero wrong answers cannot be guaranteed by any LLM system, so the system must make failures **rare, visible and non-silent**:

| Risk | Required safeguard |
|---|---|
| Invented facts | Evidence-only prompt, temperature 0, `answer_found` flag, JSON schema |
| Fabricated citations | Citation IDs validated against retrieved evidence |
| Answering when nothing relevant exists | Evidence gate + "Not found in the documents" state + 5 unanswerable eval questions |
| CLIP missing charts/diagrams | Page-linked image retrieval + vector-chart page-render fallback |
| Garbled tables | Markdown conversion + manual spot-check in Phase 2 |
| Gemini rate limit / outage | Retry with backoff, answer cache, evidence still shown |
| Demo-day failure | Cache pre-warmed for demo questions, curated example chips |
| User cannot verify | Every answer shows its evidence with doc and page |

**Known limitations to state honestly in the README:** CLIP is weaker on dense charts and text-heavy diagrams (mitigated by page-linking); scanned PDFs are unsupported; tiny text inside images may be misread by the vision model.

---

## 13. Git and GitHub Rules

- Public repo. **Ask me for the repo name and username** in Phase 0.
- One commit per phase (messages above). Commit only working states. Use `feat:`, `fix:`, `docs:`, `chore:` prefixes.
- Before every commit: `git status` and confirm only Allowed-File-Map files are present.
- **Never commit** `.env`, `data/pdfs/`, `data/index/`, `data/images/`, `data/cache.json`, `.venv/`, font files. If a secret is ever committed, tell me immediately so I can revoke the key.
- Push after each phase (`git push -u origin main` the first time). Tag `v1.0` at the end.

---

## 14. Final Definition of Done

- [ ] Only the files in the Allowed File Map exist; code is small, readable and documented
- [ ] `python ingest.py` builds both indexes from the 12 PDFs
- [ ] `python generator.py "..." --mode <mode>` works for all three modes
- [ ] `python evaluate.py` outputs the comparison table, chart and `results.json`
- [ ] Website runs with `uvicorn app:app`, supports Text-only / Text + Tables / Multimodal / Compare, uses the specified palette and fonts, shows evidence and citations, handles "not found" and errors
- [ ] README complete (setup, corpus table, results, 3 failure cases, limitations, screenshots, font credits)
- [ ] Clean-clone test passed; no secrets or PDFs tracked
- [ ] Everything pushed to the public GitHub repo and tagged `v1.0`
