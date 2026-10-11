"""
generator.py — Grounded generation with Gemini

Retrieves evidence via retriever.py, builds a multimodal prompt
(text + table markdown + image bytes), calls Gemini (temperature 0,
JSON schema output), validates citations, and caches responses.

Usage:
    python generator.py "What is Infosys revenue?" --mode multimodal
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

import config
import retriever

load_dotenv()

# --- 1. Gemini client ---

def _get_client() -> genai.Client:
    """Return a Gemini client using the API key from the environment."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("ERROR: GEMINI_API_KEY not found. Create a .env file with your key.")
        raise SystemExit(1)
    return genai.Client(api_key=key)


# --- 2. Prompt construction ---

SYSTEM_INSTRUCTION = (
    "You are a document QA assistant. Answer ONLY from the evidence provided below. "
    "If the evidence does not contain the answer, set answer_found to false and explain "
    "that the answer is not in the documents. Cite evidence IDs (e.g. E1, E2). "
    "Never invent numbers; quote figures exactly as shown. "
    "Refer to a table or figure by its page."
)

ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING"},
        "answer_found": {"type": "BOOLEAN"},
        "citations": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["answer", "answer_found", "citations"],
}


def _build_text_evidence(evidence: dict) -> list:
    """Build text/table parts of the prompt from evidence."""
    parts = []
    for item in evidence.get("texts", []) + evidence.get("tables", []):
        eid = item["evidence_id"]
        parts.append(
            f"\n[{eid}] ({item['type']}) {item['doc']} · p.{item['page']}:\n{item['text']}"
        )
    return parts


def _build_image_evidence(evidence: dict) -> list:
    """Build image parts (text prefix + bytes) of the prompt."""
    parts = []
    for item in evidence.get("images", []):
        eid = item["evidence_id"]
        prefix = f"\nImage {eid} ({item['doc']}, page {item['page']})"
        if item.get("caption"):
            prefix += f" Caption: {item['caption']}"
        prefix += ":"
        parts.append(prefix)

        img_path = config.BASE_DIR / item["path"]
        with open(img_path, "rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))
    return parts


def _build_prompt_contents(question: str, evidence: dict) -> list:
    """Assemble the full Gemini prompt from evidence and question."""
    contents = [SYSTEM_INSTRUCTION, "", "--- EVIDENCE ---"]
    contents.extend(_build_text_evidence(evidence))
    contents.extend(_build_image_evidence(evidence))
    contents.append(f"\n--- QUESTION ---\n{question}")
    return contents


# --- 3. Cache ---

def _load_cache() -> dict:
    """Load the answer cache from disk."""
    if config.CACHE_PATH.exists():
        with open(config.CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    """Write the answer cache to disk."""
    config.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


def _cache_key(question: str, mode: str) -> str:
    """Deterministic cache key from mode and normalised question."""
    return f"{mode}::{question.strip().lower()}"


def _cache_result(cache: dict, key: str, result: dict) -> dict:
    """Save a result to the cache and return it."""
    cache[key] = result
    _save_cache(cache)
    return result


# --- 4. Gemini call with retry ---

RETRYABLE_CODES = ("429", "500", "502", "503", "504")


def _call_gemini(client: genai.Client, contents: list) -> dict:
    """Call Gemini with exponential backoff on rate-limit or server errors."""
    last_error = None
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=ANSWER_SCHEMA,
                ),
            )
            return json.loads(resp.text)
        except Exception as e:
            last_error = e
            error_str = str(e)
            if any(code in error_str for code in RETRYABLE_CODES):
                wait = config.RETRY_BACKOFF_FACTOR ** attempt
                print(f"  Retry {attempt + 1}/{config.MAX_RETRIES} in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    raise last_error


# --- 5. Citation validation ---

def _validate_citations(result: dict, evidence_ids: set) -> dict:
    """Drop citation IDs that are not in the provided evidence."""
    result["citations"] = [c for c in result.get("citations", []) if c in evidence_ids]
    return result


def _validate_and_retry(client: genai.Client, contents: list,
                        result: dict, evidence_ids: set) -> dict:
    """Validate citations; retry once if answer_found but no valid citation remains."""
    result = _validate_citations(result, evidence_ids)
    if result.get("answer_found") and not result["citations"]:
        print("  No valid citations found, retrying once...")
        try:
            result = _call_gemini(client, contents)
            result = _validate_citations(result, evidence_ids)
            if result.get("answer_found") and not result["citations"]:
                result["answer"] += " [unverified: no valid citation]"
        except Exception as e:
            print(f"  Retry API Error: {e}")
    return result


# --- 6. Public API ---

def generate(question: str, mode: str = "text") -> dict:
    """Generate a grounded answer for the question using the specified mode."""
    cache = _load_cache()
    key = _cache_key(question, mode)
    if key in cache:
        print("  (served from cache)")
        cached = cache[key]
        cached["from_cache"] = True
        return cached

    evidence = retriever.search(question, mode=mode)

    if evidence["insufficient"]:
        return _cache_result(cache, key, {
            "answer": "The answer was not found in the documents.",
            "answer_found": False, "citations": [],
            "evidence": evidence, "from_cache": False,
        })

    contents = _build_prompt_contents(question, evidence)
    client = _get_client()
    all_ev = evidence.get("texts", []) + evidence.get("tables", []) + evidence.get("images", [])
    evidence_ids = {item["evidence_id"] for item in all_ev}

    start = time.time()
    try:
        result = _call_gemini(client, contents)
    except Exception as e:
        print(f"  API Error: {e}")
        return {"answer": f"Error calling Gemini: {e}", "answer_found": False,
                "citations": [], "evidence": evidence, "from_cache": False}

    latency_ms = int((time.time() - start) * 1000)
    result = _validate_and_retry(client, contents, result, evidence_ids)
    result["evidence"] = evidence
    result["latency_ms"] = latency_ms
    result["from_cache"] = False

    return _cache_result(cache, key, result)


# --- 7. CLI ---

def main() -> None:
    """CLI entry point: ask a question and print the answer."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Ask a question over your PDFs")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--mode", default="text", choices=["text", "text+tables", "multimodal"],
        help="Retrieval mode (default: text)"
    )
    args = parser.parse_args()

    print(f"Question: {args.question}")
    print(f"Mode: {args.mode}\n")

    result = generate(args.question, mode=args.mode)
    _print_result(result)


def _print_result(result: dict) -> None:
    """Print the answer, citations, and evidence summary."""
    print(f"\nAnswer: {result.get('answer', '')}")
    print(f"Found: {result.get('answer_found', False)}")
    print(f"Citations: {result.get('citations', [])}")
    if "latency_ms" in result:
        print(f"Latency: {result['latency_ms']} ms")

    ev = result.get("evidence", {})
    all_ev = ev.get("texts", []) + ev.get("tables", []) + ev.get("images", [])
    all_ev.sort(key=lambda x: int(x["evidence_id"][1:]))

    if all_ev:
        print("\n--- Evidence ---")
        for item in all_ev:
            eid = item["evidence_id"]
            print(f"  {eid} ({item['type']}): {item['doc']} p.{item['page']} (score={item['score']})")


if __name__ == "__main__":
    main()
