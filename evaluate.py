"""
evaluate.py — Evaluation across retrieval modes

Runs every question in data/eval_questions.json through each mode
(text, text+tables, multimodal) and computes:
  - Page hit@k (deterministic, no LLM)
  - Answer correctness (Gemini-as-judge)
  - Abstention accuracy
  - RAGAS faithfulness & relevancy (Gemini-as-judge, time-boxed)
  - Latency per query

Outputs: data/results.json, a printed summary table, and data/results.png.

Usage:
    python evaluate.py
"""

import json
import time
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai
from google.genai import types

import config
import generator

load_dotenv()


# --- 1. Gemini client ---

def _get_eval_client() -> genai.Client:
    """Return a Gemini client for evaluation judging."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("ERROR: GEMINI_API_KEY not found.")
        raise SystemExit(1)
    return genai.Client(api_key=key)


# --- 2. LLM judges ---

def judge_correctness(client: genai.Client, question: str,
                      gold: str, generated: str) -> str:
    """Compare generated answer to gold answer; return CORRECT, PARTIAL, or WRONG."""
    prompt = (
        "You are an impartial judge evaluating a QA system.\n"
        f"Question: {question}\n"
        f"Gold Answer: {gold}\n"
        f"Generated Answer: {generated}\n\n"
        "Compare the Generated Answer to the Gold Answer. Respond with EXACTLY ONE WORD:\n"
        "- CORRECT: if they match or are functionally equivalent\n"
        "- PARTIAL: if the generated answer is incomplete or has minor errors\n"
        "- WRONG: if they contradict or are completely unrelated"
    )
    for _ in range(config.MAX_RETRIES):
        time.sleep(config.EVAL_CALL_DELAY)
        try:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL, contents=[prompt],
                config=types.GenerateContentConfig(temperature=0),
            )
            text = resp.text.strip().upper()
            if "CORRECT" in text:
                return "CORRECT"
            if "PARTIAL" in text:
                return "PARTIAL"
            return "WRONG"
        except Exception as e:
            print(f"    Judge call failed: {str(e)[:80]}")
    return "WRONG"


def judge_ragas(client: genai.Client, question: str, answer: str,
                evidence_texts: list[str]) -> tuple[int, int]:
    """Return (faithfulness, relevancy) each 0 or 1 via Gemini-as-judge."""
    ev_str = "\n".join(evidence_texts)
    prompt = (
        "Evaluate the following answer based on the evidence provided and the question asked.\n"
        f"Question: {question}\n"
        f"Evidence: {ev_str}\n"
        f"Answer: {answer}\n\n"
        "Return a JSON object with two integer fields (1 for Yes/Good, 0 for No/Bad):\n"
        "- faithfulness: 1 if the answer is completely derived from the evidence, 0 if it contains fabricated facts.\n"
        "- relevancy: 1 if the answer directly and concisely addresses the question, 0 otherwise."
    )
    schema = {
        "type": "OBJECT",
        "properties": {
            "faithfulness": {"type": "INTEGER"},
            "relevancy": {"type": "INTEGER"},
        },
    }
    for _ in range(config.MAX_RETRIES):
        time.sleep(config.EVAL_CALL_DELAY)
        try:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL, contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            data = json.loads(resp.text)
            return data.get("faithfulness", 0), data.get("relevancy", 0)
        except Exception as e:
            print(f"    Judge call failed: {str(e)[:80]}")
    return 0, 0


# --- 3. Per-question evaluation ---

def _evaluate_answerable(client: genai.Client, q: dict,
                         ans_data: dict) -> dict:
    """Evaluate a single answerable question and return metrics."""
    gold_doc = q.get("gold_doc")
    gold_page = q.get("gold_page")
    all_ev = ans_data.get("evidence", {})
    all_items = all_ev.get("texts", []) + all_ev.get("tables", []) + all_ev.get("images", [])

    hit = any(i.get("doc") == gold_doc and i.get("page") == gold_page for i in all_items)
    ev_texts = [i["text"] for i in all_items if "text" in i]
    ev_texts += [i["caption"] for i in all_items if "caption" in i and i["caption"]]

    ans_found = ans_data.get("answer_found", False)
    correctness, faithfulness, relevancy = "WRONG", 0, 0

    if ans_found:
        correctness = judge_correctness(
            client, q["question"], q.get("gold_answer", ""), ans_data.get("answer", "")
        )
        try:
            faithfulness, relevancy = judge_ragas(
                client, q["question"], ans_data.get("answer", ""), ev_texts
            )
        except Exception as e:
            print(f"    RAGAS failed: {e}")

    return {
        "id": q["id"], "hit": hit, "false_abstention": not ans_found,
        "correctness": correctness, "faithfulness": faithfulness,
        "relevancy": relevancy, "latency": ans_data.get("latency_ms", 0),
    }


# --- 4. Summary and plotting ---

def _print_summary(results: dict, modes: list[str]) -> None:
    """Print a markdown summary table of evaluation results."""
    print("\n### Evaluation Summary")
    print("| Mode | Hit@k | Correct | False Abstain | True Abstain "
          "| Faithfulness | Relevancy | Avg Latency |")
    print("|---|---|---|---|---|---|---|---|")

    for mode in modes:
        ans = results[mode]["answerable"]
        unans = results[mode]["unanswerable"]
        n_ans = len(ans) if ans else 1
        n_unans = len(unans) if unans else 1

        hit_pct = sum(1 for x in ans if x["hit"]) / n_ans * 100
        corr_pct = sum(1 for x in ans if x["correctness"] == "CORRECT") / n_ans * 100
        false_abs = sum(1 for x in ans if x["false_abstention"])
        true_abs = sum(1 for x in unans if x["correct_abstention"]) / n_unans * 100
        faith = sum(x["faithfulness"] for x in ans) / n_ans * 100
        rel = sum(x["relevancy"] for x in ans) / n_ans * 100
        lats = [x["latency"] for x in ans] + [x["latency"] for x in unans]
        avg_lat = sum(lats) / len(lats) if lats else 0

        print(f"| {mode} | {hit_pct:.1f}% | {corr_pct:.1f}% | {false_abs} "
              f"| {true_abs:.1f}% | {faith:.1f}% | {rel:.1f}% | {avg_lat:.0f}ms |")


def plot_results(results: dict) -> None:
    """Save a bar chart comparing correctness and hit@k across modes."""
    modes = list(results.keys())
    acc_scores, hit_scores = [], []

    for m in modes:
        ans = results[m]["answerable"]
        total = len(ans) if ans else 1
        acc_scores.append(sum(1 for x in ans if x["correctness"] == "CORRECT") / total * 100)
        hit_scores.append(sum(1 for x in ans if x["hit"]) / total * 100)

    x = range(len(modes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([i - width / 2 for i in x], acc_scores, width, label="Correctness (%)")
    ax.bar([i + width / 2 for i in x], hit_scores, width, label="Hit@k (%)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Evaluation Results by Mode")
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes)
    ax.legend()

    out_path = config.DATA_DIR / "results.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved plot to {out_path}")


# --- 5. Main ---

def main() -> None:
    """Run the full evaluation pipeline."""
    q_file = config.DATA_DIR / "eval_questions.json"
    if not q_file.exists():
        with open(q_file, "w") as f:
            f.write("[]")

    with open(q_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not questions:
        print(f"Please populate {q_file} with questions before running evaluation.")
        return

    client = _get_eval_client()
    modes = ["text", "text+tables", "multimodal"]
    final = {m: {"answerable": [], "unanswerable": []} for m in modes}

    print(f"Starting evaluation of {len(questions)} questions across {len(modes)} modes...\n")

    for mode in modes:
        print(f"--- Evaluating Mode: {mode} ---")
        for q in questions:
            print(f"  Q: {q.get('question', '')[:50]}...")

            ans_data = generator.generate(q["question"], mode=mode)
            if not ans_data.get("from_cache", False):
                time.sleep(config.EVAL_CALL_DELAY)

            if q.get("type") == "unanswerable":
                final[mode]["unanswerable"].append({
                    "id": q["id"],
                    "correct_abstention": not ans_data.get("answer_found", False),
                    "latency": ans_data.get("latency_ms", 0),
                })
            else:
                final[mode]["answerable"].append(
                    _evaluate_answerable(client, q, ans_data)
                )

    with open(config.RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    _print_summary(final, modes)
    plot_results(final)


if __name__ == "__main__":
    main()
