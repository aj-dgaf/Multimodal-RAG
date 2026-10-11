# How to Start the Project

All commands are for Windows PowerShell and are run from the project folder (the one containing `app.py`).

## 1. One-time setup

1. Install Python 3.11.

2. Create and activate a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```
   If activation is blocked, run this once and try again:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```

3. Install the dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

4. Add your Gemini API key. Copy the example file and replace `your_key_here` with a key from Google AI Studio:
   ```powershell
   copy .env.example .env
   notepad .env
   ```

5. Put the 12 PDFs listed in the README corpus table into `data\pdfs\`. They are not in the repository.

6. Build the search indexes. This reads every PDF, extracts text, tables and images, and takes several minutes. The first run also downloads the two embedding models.
   ```powershell
   python ingest.py
   ```
   It creates `data\index\` and `data\images\`. Run it again only if the PDFs change.

## 2. Start the website

```powershell
.venv\Scripts\Activate.ps1
uvicorn app:app --reload
```

Wait for `Application startup complete`, then open http://127.0.0.1:8000 in a browser. Press `Ctrl+C` in the terminal to stop the server.

## 3. Ask a question from the terminal (optional)

```powershell
python generator.py "State the two laws of reflection of light." --mode multimodal
```

`--mode` can be `text`, `text+tables` or `multimodal`.

## 4. Run the evaluation (optional)

```powershell
python evaluate.py
```

This runs the 35 questions in `data\eval_questions.json` through all three modes and writes `data\results.json` and `data\results.png`. A full run takes about 30 minutes because calls are spaced 5 seconds apart to stay under the free-tier limit. Answers are cached, so a second run is faster.

## Demo tips

- `question.txt` lists questions that are already cached. They answer instantly and use no API quota.
- Use **Compare** mode with the first four questions in that file to show text-only failing where multimodal succeeds.

## If something goes wrong

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY not found` | The `.env` file is missing or empty. Redo setup step 4. |
| `Text index not found. Run ingest.py first.` | Redo setup step 6. |
| Answer says `Error calling Gemini: 429 ...` | The free-tier limit was hit (15 requests per minute, plus a daily cap). Wait a minute and ask again, or use a cached question. |
| `uvicorn` is not recognised | The virtual environment is not active. Run `.venv\Scripts\Activate.ps1` first. |
| Port 8000 is already in use | Start on another port: `uvicorn app:app --port 8001`. |
| No images appear in answers | `data\images\` is missing. Run `python ingest.py`. |
