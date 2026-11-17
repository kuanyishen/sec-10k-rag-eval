"""Answer every question under several retrieval arms with a small model, then grade.

Numeric and unanswerable questions are graded programmatically. Text answers are graded by an
LLM judge (judge.py), which is itself validated on the numeric answers where the truth is known.

    python -m rag_eval.run_generation        (needs ANTHROPIC_API_KEY; runs on GitHub Actions)
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import generate as G, llm, metrics as M
from .chunk import ChunkCfg
from .corpus import build, load_questions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "generation"
CFG = ChunkCfg(400, 50, True)          # the primary chunking, fixed before the experiment
GEN_MODEL = "claude-haiku-5-5"
ARMS = {
    "closed_book": {"ret": None},
    "bm25_k5": {"ret": "bm25", "k": 5},
    "hybrid_k5": {"ret": "hybrid", "k": 5},
    "rerank_k3": {"ret": "hybrid_rerank", "k": 3},
    "rerank_k5": {"ret": "hybrid_rerank", "k": 5},
    "rerank_k10": {"ret": "hybrid_rerank", "k": 10},
    "oracle_k5": {"ret": "hybrid_rerank", "k": 5, "oracle": True},
}


def contexts_for(arm: dict, q: dict, runs: dict, chunks_by_id: dict, hit_ids: dict) -> list[str] | None:
    if arm["ret"] is None:
        return None
    ids = runs[arm["ret"]][q["id"]][: arm["k"]]
    if arm.get("oracle"):
        gold = hit_ids.get(q["id"])
        if gold:  # put a known answer-bearing chunk first, keep the rest as realistic distractors
            ids = [gold] + [i for i in ids if i != gold][: arm["k"] - 1]
    return [chunks_by_id[i] for i in ids]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qs = load_questions(("numeric", "text", "unanswerable"))
    chunks = build(CFG)
    by_id = {c.chunk_id: c for c in chunks}
    runs = {r: {x["id"]: x["top"] for x in json.loads((ROOT / "results" / "retrieval" / f"{CFG.name}__{r}.json").read_text())}
            for r in ("bm25", "hybrid", "hybrid_rerank")}
    # an answer-bearing chunk per answerable question: prefer one the reranker ranked, else the first in the filing
    hit_ids = {}
    for q in qs:
        if q["type"] == "unanswerable":
            continue
        hits = [c.chunk_id for c in chunks if M.is_hit(c.ticker, c.body, q)]
        ranked = [i for i in runs["hybrid_rerank"].get(q["id"], []) if i in set(hits)]
        hit_ids[q["id"]] = (ranked or hits or [None])[0]

    def answer(job):
        arm_name, q = job
        ctx = contexts_for(ARMS[arm_name], q, runs, by_id, hit_ids)
        texts = None if ctx is None else [c.text for c in ctx]
        system, user = G.build_prompt(q, texts)
        try:
            a = llm.claude(user, system=system, model=GEN_MODEL, max_tokens=1500, effort="low")
        except Exception as e:
            a = f"ERROR: {str(e)[:80]}"
        row = {"id": q["id"], "type": q["type"], "ticker": q["ticker"], "arm": arm_name, "answer": a.strip(),
               "abstained": G.abstained(a), "contexts": [] if ctx is None else [c.chunk_id for c in ctx],
               "gold_in_context": None if ctx is None or q["type"] == "unanswerable"
               else any(M.is_hit(c.ticker, c.body, q) for c in ctx)}
        if q["type"] == "numeric":
            row["correct"] = G.numeric_correct(a, q["answer_value"], q["unit"])
            row["number_in_context"] = False if ctx is None else G.number_in_context(q, [c.body for c in ctx])
        elif q["type"] == "unanswerable":
            row["correct"] = row["abstained"]
        return row

    jobs = [(a, q) for a in ARMS for q in qs]
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(answer, jobs))
    for a in ARMS:
        (OUT / f"{a}.jsonl").write_text("\n".join(json.dumps(r) for r in rows if r["arm"] == a) + "\n")
        sub = [r for r in rows if r["arm"] == a]
        num = [r for r in sub if r["type"] == "numeric"]
        una = [r for r in sub if r["type"] == "unanswerable"]
        print(f"{a:12s} numeric correct {sum(r['correct'] for r in num)}/{len(num)}  "
              f"abstain on unanswerable {sum(r['correct'] for r in una)}/{len(una)}  "
              f"errors {sum(r['answer'].startswith('ERROR') for r in sub)}")
    print("usage:", llm.USAGE)


if __name__ == "__main__":
    main()
