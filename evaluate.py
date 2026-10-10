"""
evaluate.py — Phase 4: Evaluation

Evaluates the RAG system across text, text+tables, and multimodal modes.
Metrics:
- Page hit@k
- Answer correctness (Gemini as judge)
- Abstention accuracy
- RAGAS (faithfulness & relevancy via Gemini)
- Latency

Usage:
    python evaluate.py
"""

import json
import time
import os
import matplotlib.pyplot as plt
from google import genai
from google.genai import types

import config
import generator

def _get_eval_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("ERROR: GEMINI_API_KEY not found.")
        raise SystemExit(1)
    return genai.Client(api_key=key)

def judge_correctness(client: genai.Client, question: str, gold: str, generated: str) -> str:
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
        try:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0)
            )
            text = resp.text.strip().upper()
            if "CORRECT" in text: return "CORRECT"
            if "PARTIAL" in text: return "PARTIAL"
            return "WRONG"
        except Exception as e:
            time.sleep(2)
    return "WRONG"

def judge_ragas(client: genai.Client, question: str, answer: str, evidence_texts: list[str]) -> tuple[int, int]:
    # Simplified RAGAS metrics via LLM
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
            "relevancy": {"type": "INTEGER"}
        }
    }
    for _ in range(config.MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=schema
                )
            )
            data = json.loads(resp.text)
            return data.get("faithfulness", 0), data.get("relevancy", 0)
        except Exception as e:
            time.sleep(2)
    return 0, 0

def plot_results(results: dict):
    modes = list(results.keys())
    # Extract correctness (we consider CORRECT as 1, PARTIAL as 0.5 for a general accuracy score)
    # Alternatively, just plot "CORRECT %"
    acc_scores = []
    hit_scores = []
    for m in modes:
        res_m = results[m]
        total_ans = len(res_m["answerable"])
        if total_ans == 0:
            acc_scores.append(0)
            hit_scores.append(0)
            continue
            
        corrects = sum(1 for x in res_m["answerable"] if x["correctness"] == "CORRECT")
        acc_scores.append(corrects / total_ans * 100)
        
        hits = sum(1 for x in res_m["answerable"] if x["hit"])
        hit_scores.append(hits / total_ans * 100)
        
    x = range(len(modes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([i - width/2 for i in x], acc_scores, width, label='Correctness (%)')
    ax.bar([i + width/2 for i in x], hit_scores, width, label='Hit@k (%)')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Evaluation Results by Mode')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    
    out_path = config.DATA_DIR / "results.png"
    plt.savefig(out_path)
    print(f"\nSaved plot to {out_path}")

def main():
    q_file = config.DATA_DIR / "eval_questions.json"
    if not q_file.exists():
        with open(q_file, "w") as f:
            f.write("[]")
            
    with open(q_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    if not questions:
        print(f"Please populate {q_file} with questions before running evaluation.")
        print('Format: [{"id": "q1", "question": "...", "type": "text", "gold_answer": "...", "gold_doc": "doc.pdf", "gold_page": 1}]')
        return

    client = _get_eval_client()
    modes = ["text", "text+tables", "multimodal"]
    final_results = {m: {"answerable": [], "unanswerable": []} for m in modes}
    
    print(f"Starting evaluation of {len(questions)} questions across {len(modes)} modes...\n")
    
    for mode in modes:
        print(f"--- Evaluating Mode: {mode} ---")
        for q in questions:
            q_id = q.get("id")
            q_text = q.get("question")
            q_type = q.get("type", "text")
            is_unanswerable = (q_type == "unanswerable")
            
            print(f"  Q: {q_text[:50]}...")
            
            # Generate answer (uses cache if available)
            ans_data = generator.generate(q_text, mode=mode)
            
            # Sleep slightly if it wasn't cached to avoid rate limits
            if not ans_data.get("from_cache", False):
                time.sleep(1)
                
            ans_found = ans_data.get("answer_found", False)
            latency = ans_data.get("latency_ms", 0)
            
            # Evaluate Hit@k
            hit = False
            ev_texts = []
            if not is_unanswerable:
                gold_doc = q.get("gold_doc")
                gold_page = q.get("gold_page")
                all_ev = ans_data.get("evidence", {})
                all_items = all_ev.get("texts", []) + all_ev.get("tables", []) + all_ev.get("images", [])
                
                for item in all_items:
                    if item.get("doc") == gold_doc and item.get("page") == gold_page:
                        hit = True
                    if "text" in item:
                        ev_texts.append(item["text"])
                    elif "caption" in item:
                        ev_texts.append(item["caption"])

            if is_unanswerable:
                final_results[mode]["unanswerable"].append({
                    "id": q_id,
                    "correct_abstention": not ans_found,
                    "latency": latency
                })
            else:
                correctness = "WRONG"
                faithfulness = 0
                relevancy = 0
                
                if ans_found:
                    gold_ans = q.get("gold_answer", "")
                    gen_ans = ans_data.get("answer", "")
                    correctness = judge_correctness(client, q_text, gold_ans, gen_ans)
                    
                    try:
                        faithfulness, relevancy = judge_ragas(client, q_text, gen_ans, ev_texts)
                    except Exception as e:
                        print(f"    RAGAS failed: {e}")
                
                final_results[mode]["answerable"].append({
                    "id": q_id,
                    "hit": hit,
                    "false_abstention": not ans_found,
                    "correctness": correctness,
                    "faithfulness": faithfulness,
                    "relevancy": relevancy,
                    "latency": latency
                })
                
    # Save results
    with open(config.RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
        
    # Print Markdown Summary
    print("\n### Evaluation Summary")
    print("| Mode | Hit@k | Correct | False Abstain | True Abstain | Faithfulness | Relevancy | Avg Latency |")
    print("|---|---|---|---|---|---|---|---|")
    
    for mode in modes:
        ans = final_results[mode]["answerable"]
        unans = final_results[mode]["unanswerable"]
        
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
        
        print(f"| {mode} | {hit_pct:.1f}% | {corr_pct:.1f}% | {false_abs} | {true_abs:.1f}% | {faith:.1f}% | {rel:.1f}% | {avg_lat:.0f}ms |")
        
    plot_results(final_results)

if __name__ == "__main__":
    main()
