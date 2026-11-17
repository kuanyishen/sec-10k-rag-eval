"""Grade free-text answers with the LLM judge, and measure how far the judge can be trusted.

    python -m rag_eval.run_judge        (needs ANTHROPIC_API_KEY; runs after run_generation)
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import judge as J, llm
from .corpus import build, load_questions
from .run_generation import ARMS, CFG

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "results" / "generation"
OUT = ROOT / "results" / "judge"


def ref_text(q: dict) -> str:
    if q["type"] == "numeric":
        v = q["answer_value"]
        return f"${v:,.2f} per share" if q["unit"] == "dollars per share" else f"${v:,.0f} million"
    return q["answer"]


def make_probes(num_rows: list[dict], qs: dict, rng: random.Random, n: int = 60) -> list:
    """Correct numeric answers with the value changed by 10-30% everywhere: a judge must call these incorrect."""
    probes = []
    for r in [r for r in num_rows if r["correct"]]:
        q = qs[r["id"]]
        c = J.corrupt_answer(r["answer"], q["answer_value"], q["unit"], rng.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3]))
        if c:
            probes.append(("val_corrupt", r, c, J.JUDGE_MODEL))
        if len(probes) == n:
            break
    return probes


def probes_only():
    """Re-run just the corrupted-answer check and update judge_validation.json."""
    qs = {q["id"]: q for q in load_questions(("numeric", "text", "unanswerable"))}
    by_id = {c.chunk_id: c for c in build(CFG)}
    rows = [json.loads(l) for a in ARMS for l in (GEN / f"{a}.jsonl").read_text().splitlines() if l.strip()]
    num = [r for r in rows if r["type"] == "numeric" and r["arm"] in ("dense_k5", "bm25_k5", "closed_book")]
    probes = make_probes(num, qs, random.Random(1))

    def run(item):
        _, r, answer, model = item
        q = qs[r["id"]]
        v = J.judge(q["question"], ref_text(q), [by_id[c].text for c in r["contexts"]], answer, model=model)
        return {"tag": "val_corrupt_v2", "id": r["id"], "arm": r["arm"], "answer": answer, "verdict": v}

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(run, probes))
    with (OUT / "verdicts.jsonl").open("a") as f:
        f.write("\n".join(json.dumps(x) for x in res) + "\n")
    p = ROOT / "results" / "judge_validation.json"
    summary = json.loads(p.read_text())
    ok = [x for x in res if x["verdict"]]
    summary["corrupted_answers_rejected_v1_buggy_probe"] = summary.pop("corrupted_answers_rejected", None)
    summary["corrupted_answers_rejected"] = {"n": len(ok), "rejected": sum(not x["verdict"]["correct"] for x in ok)}
    p.write_text(json.dumps(summary, indent=1))
    print(summary["corrupted_answers_rejected"], "usage:", llm.USAGE)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qs = {q["id"]: q for q in load_questions(("numeric", "text", "unanswerable"))}
    by_id = {c.chunk_id: c for c in build(CFG)}
    rows = [json.loads(l) for a in ARMS for l in (GEN / f"{a}.jsonl").read_text().splitlines() if l.strip()]

    def run(item):
        tag, r, answer, model = item
        q = qs[r["id"]]
        try:
            v = J.judge(q["question"], ref_text(q), [by_id[c].text for c in r["contexts"]], answer, model=model)
        except Exception as e:
            v = None
        return {"tag": tag, "id": r["id"], "arm": r["arm"], "model": model, "verdict": v}

    rng = random.Random(0)
    items = [("text", r, r["answer"], J.JUDGE_MODEL) for r in rows if r["type"] == "text"]
    # validation 1: numeric answers, where correctness and support are known programmatically
    num = [r for r in rows if r["type"] == "numeric" and r["arm"] in ("dense_k5", "bm25_k5", "closed_book") and not r["answer"].startswith("ERROR")]
    rng.shuffle(num)
    val = num[:180]
    items += [("val_numeric", r, r["answer"], J.JUDGE_MODEL) for r in val]
    # validation 2: correct answers with the number deliberately changed -> must be judged incorrect
    items += make_probes(num, qs, rng)
    # second opinion: a stronger model re-judges a sample of the primary arm's text answers
    prim = [r for r in rows if r["type"] == "text" and r["arm"] == "dense_k5"]
    items += [("second_opinion", r, r["answer"], J.CHECK_MODEL) for r in rng.sample(prim, min(60, len(prim)))]

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(run, items))
    (OUT / "verdicts.jsonl").write_text("\n".join(json.dumps(x) for x in res) + "\n")

    # --- judge validation summary -------------------------------------------------------
    rmap = {(r["id"], r["arm"]): r for r in rows}
    vnum = [x for x in res if x["tag"] == "val_numeric" and x["verdict"]]
    corr_pairs = [(x["verdict"]["correct"], rmap[(x["id"], x["arm"])]["correct"]) for x in vnum]
    sup_pairs = [(x["verdict"]["supported"],
                  rmap[(x["id"], x["arm"])]["abstained"] or rmap[(x["id"], x["arm"])]["number_in_context"])
                 for x in vnum if x["arm"] != "closed_book"]
    vcor = [x for x in res if x["tag"] == "val_corrupt" and x["verdict"]]
    text_h = {(x["id"], x["arm"]): x["verdict"] for x in res if x["tag"] == "text" and x["verdict"]}
    so = [(x["verdict"]["correct"], text_h[(x["id"], x["arm"])]["correct"]) for x in res
          if x["tag"] == "second_opinion" and x["verdict"] and (x["id"], x["arm"]) in text_h]
    summary = {
        "judge_model": J.JUDGE_MODEL, "check_model": J.CHECK_MODEL,
        "numeric_correct_vs_truth": J.agreement(corr_pairs),
        "numeric_supported_vs_truth": J.agreement(sup_pairs),
        "corrupted_answers_rejected": {"n": len(vcor), "rejected": sum(not x["verdict"]["correct"] for x in vcor)},
        "second_opinion_correct_agreement": J.agreement(so),
        "unparsed": sum(x["verdict"] is None for x in res),
    }
    (ROOT / "results" / "judge_validation.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("usage:", llm.USAGE)


if __name__ == "__main__":
    import sys
    probes_only() if "--probes" in sys.argv else main()
