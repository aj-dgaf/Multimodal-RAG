"""
generator.py — Phase 1: Text-only generation

Retrieves evidence via retriever.py, builds a grounded prompt, calls
Gemini (temperature 0, JSON schema output), validates citations, and
caches responses.

Phase 3 will extend this to support multimodal evidence (images).

Usage:
    python generator.py "What is Infosys revenue?" --mode text
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
    """Create a Gemini API client from the environment key."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("ERROR: GEMINI_API_KEY not found. Create a .env file with your key.")
        raise SystemExit(1)
    return genai.Client(api_key=key)


# --- 2. Build the prompt ---

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


def _build_prompt(question: str, evidence: dict) -> str:
    """Build a text prompt from retrieved evidence."""
    parts = [SYSTEM_INSTRUCTION, "", "--- EVIDENCE ---"]

    for item in evidence["texts"]:
        eid = item["evidence_id"]
        doc = item["doc"]
        page = item["page"]
        etype = item["type"]
        text = item["text"]
        parts.append(f"\n[{eid}] ({etype}) {doc} · p.{page}:\n{text}")

    parts.append(f"\n--- QUESTION ---\n{question}")
    return "\n".join(parts)


# --- 3. Cache ---

def _load_cache() -> dict:
    """Load the answer cache from disk."""
    if config.CACHE_PATH.exists():
        with open(config.CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    """Save the answer cache to disk."""
    config.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


def _cache_key(question: str, mode: str) -> str:
    """Normalised cache key: lowercase, stripped, mode-prefixed."""
    return f"{mode}::{question.strip().lower()}"


# --- 4. Call Gemini with retry ---

def _call_gemini(client: genai.Client, prompt: str) -> dict:
    """Call Gemini with retry on transient errors. Returns parsed JSON."""
    last_error = None
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[prompt],
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
            # Retry on rate limit or server errors
            if "429" in error_str or "5" in error_str[:1]:
                wait = config.RETRY_BACKOFF_FACTOR ** attempt
                print(f"  Retry {attempt + 1}/{config.MAX_RETRIES} in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    raise last_error


# --- 5. Validate citations ---

def _validate_citations(result: dict, evidence_ids: set) -> dict:
    """Drop citation IDs that are not in the provided evidence."""
    valid = [c for c in result.get("citations", []) if c in evidence_ids]
    result["citations"] = valid
    return result


# --- 6. Generate answer ---

def generate(question: str, mode: str = "text") -> dict:
    """Full pipeline: retrieve, build prompt, call Gemini, validate, cache."""
    # Check cache
    cache = _load_cache()
    key = _cache_key(question, mode)
    if key in cache:
        print("  (served from cache)")
        cached = cache[key]
        cached["from_cache"] = True
        return cached

    # Retrieve evidence
    evidence = retriever.search(question, mode=mode)

    if evidence["insufficient"]:
        result = {
            "answer": "The answer was not found in the documents.",
            "answer_found": False,
            "citations": [],
            "evidence": evidence,
            "from_cache": False,
        }
        cache[key] = result
        _save_cache(cache)
        return result

    # Build prompt and call Gemini
    prompt = _build_prompt(question, evidence)
    client = _get_client()

    start = time.time()
    result = _call_gemini(client, prompt)
    latency_ms = int((time.time() - start) * 1000)

    # Validate citations
    evidence_ids = {item["evidence_id"] for item in evidence["texts"]}
    result = _validate_citations(result, evidence_ids)

    # If answer_found but no valid citations, retry once
    if result.get("answer_found") and not result["citations"]:
        print("  No valid citations found, retrying once...")
        result = _call_gemini(client, prompt)
        result = _validate_citations(result, evidence_ids)
        if result.get("answer_found") and not result["citations"]:
            result["answer"] += " [unverified: no valid citation]"

    result["evidence"] = evidence
    result["latency_ms"] = latency_ms
    result["from_cache"] = False

    # Save to cache
    cache[key] = result
    _save_cache(cache)

    return result


# --- 7. CLI ---

def main() -> None:
    """CLI entry point for testing generation."""
    parser = argparse.ArgumentParser(description="Ask a question over your PDFs")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--mode", default="text", choices=["text", "text+tables", "multimodal"],
        help="Retrieval mode (default: text)"
    )
    # Windows consoles default to cp1252 which can't print ₹, © etc.
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    args = parser.parse_args()

    print(f"Question: {args.question}")
    print(f"Mode: {args.mode}\n")

    result = generate(args.question, mode=args.mode)

    print(f"\nAnswer: {result['answer']}")
    print(f"Found: {result['answer_found']}")
    print(f"Citations: {result['citations']}")
    if "latency_ms" in result:
        print(f"Latency: {result['latency_ms']} ms")

    # Show evidence summary
    ev = result.get("evidence", {})
    for item in ev.get("texts", []):
        doc = item["doc"]
        page = item["page"]
        score = item["score"]
        eid = item["evidence_id"]
        print(f"  {eid}: {doc} p.{page} (score={score})")


if __name__ == "__main__":
    main()
