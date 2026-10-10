"""
generator.py — Phase 3: Multimodal generation

Retrieves evidence via retriever.py, builds a grounded multimodal prompt,
calls Gemini (temperature 0, JSON schema output), validates citations,
and caches responses.

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


def _get_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("ERROR: GEMINI_API_KEY not found. Create a .env file with your key.")
        raise SystemExit(1)
    return genai.Client(api_key=key)


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


def _build_prompt_contents(question: str, evidence: dict) -> list:
    """Build a multimodal prompt from retrieved evidence."""
    contents = [SYSTEM_INSTRUCTION, "", "--- EVIDENCE ---"]

    # Add text and table evidence
    for item in evidence.get("texts", []) + evidence.get("tables", []):
        eid = item["evidence_id"]
        doc = item["doc"]
        page = item["page"]
        etype = item["type"]
        text = item["text"]
        contents.append(f"\n[{eid}] ({etype}) {doc} · p.{page}:\n{text}")

    # Add image evidence
    for item in evidence.get("images", []):
        eid = item["evidence_id"]
        doc = item["doc"]
        page = item["page"]
        caption = item.get("caption", "")
        
        # Text prefix for the image
        prefix = f"\nImage {eid} ({doc}, page {page})"
        if caption:
            prefix += f" Caption: {caption}"
        prefix += ":"
        contents.append(prefix)
        
        # Read image bytes
        img_path = config.BASE_DIR / item["path"]
        with open(img_path, "rb") as f:
            img_bytes = f.read()
            
        contents.append(
            types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        )

    contents.append(f"\n--- QUESTION ---\n{question}")
    return contents


def _load_cache() -> dict:
    if config.CACHE_PATH.exists():
        with open(config.CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    config.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


def _cache_key(question: str, mode: str) -> str:
    return f"{mode}::{question.strip().lower()}"


def _call_gemini(client: genai.Client, contents: list) -> dict:
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
            if "429" in error_str or "5" in error_str[:1]:
                wait = config.RETRY_BACKOFF_FACTOR ** attempt
                print(f"  Retry {attempt + 1}/{config.MAX_RETRIES} in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    raise last_error


def _validate_citations(result: dict, evidence_ids: set) -> dict:
    valid = [c for c in result.get("citations", []) if c in evidence_ids]
    result["citations"] = valid
    return result


def generate(question: str, mode: str = "text") -> dict:
    cache = _load_cache()
    key = _cache_key(question, mode)
    if key in cache:
        print("  (served from cache)")
        cached = cache[key]
        cached["from_cache"] = True
        return cached

    evidence = retriever.search(question, mode=mode)

    # Convert all evidence lists into a flat list of dicts to extract EIDs
    all_evidence_items = evidence.get("texts", []) + evidence.get("tables", []) + evidence.get("images", [])

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

    contents = _build_prompt_contents(question, evidence)
    client = _get_client()

    start = time.time()
    try:
        result = _call_gemini(client, contents)
    except Exception as e:
        # Robustness: return evidence with friendly message
        print(f"  API Error: {e}")
        return {
            "answer": f"Error calling Gemini: {e}",
            "answer_found": False,
            "citations": [],
            "evidence": evidence,
            "from_cache": False
        }
        
    latency_ms = int((time.time() - start) * 1000)

    evidence_ids = {item["evidence_id"] for item in all_evidence_items}
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

    result["evidence"] = evidence
    result["latency_ms"] = latency_ms
    result["from_cache"] = False

    cache[key] = result
    _save_cache(cache)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over your PDFs")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--mode", default="text", choices=["text", "text+tables", "multimodal"],
        help="Retrieval mode (default: text)"
    )
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    args = parser.parse_args()

    print(f"Question: {args.question}")
    print(f"Mode: {args.mode}\n")

    result = generate(args.question, mode=args.mode)

    print(f"\nAnswer: {result.get('answer', '')}")
    print(f"Found: {result.get('answer_found', False)}")
    print(f"Citations: {result.get('citations', [])}")
    if "latency_ms" in result:
        print(f"Latency: {result['latency_ms']} ms")

    ev = result.get("evidence", {})
    all_ev = ev.get("texts", []) + ev.get("tables", []) + ev.get("images", [])
    
    # Sort evidence to display E1, E2, ... in order
    all_ev.sort(key=lambda x: int(x["evidence_id"][1:]))
    
    if all_ev:
        print("\n--- Evidence ---")
        for item in all_ev:
            doc = item["doc"]
            page = item["page"]
            score = item["score"]
            eid = item["evidence_id"]
            etype = item["type"]
            print(f"  {eid} ({etype}): {doc} p.{page} (score={score})")


if __name__ == "__main__":
    main()
