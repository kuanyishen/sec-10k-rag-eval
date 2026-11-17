"""LLM-written questions about specific passages. Each carries a verbatim evidence quote, which is
verified by substring match; questions whose quote is not in the passage are discarded.

    python -m rag_eval.gen_text_questions        (needs ANTHROPIC_API_KEY; runs on GitHub Actions)
"""
from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from . import llm
from .chunk import ChunkCfg, chunk_blocks
from .parse import blocks_from_html

ROOT = Path(__file__).resolve().parents[2]
MODEL = "claude-sonnet-5-5"
PER_COMPANY = 16
SKIP_SECTIONS = ("Front matter", "Item 15", "Item 16", "Item 6", "Item 4", "Item 9B", "Item 9C")

PROMPT = """You are writing test questions for a system that answers questions about SEC 10-K filings.

Company: {name}. Fiscal year ended {end}. Section: {section}.

Passage from the filing:
\"\"\"
{passage}
\"\"\"

Write ONE question that:
- can be answered using only this passage,
- names the company ("{name}") and does not rely on the reader knowing which passage it came from,
- is specific (a concrete fact, list, or explanation), not "summarize the section",
- does not copy the passage's wording more than necessary.

Also give the short correct answer, and the evidence: one or two consecutive sentences copied EXACTLY (character for character) from the passage that contain the answer, at most 60 words.
Classify the question as "fact" (single value or name), "list" (several items) or "explanation" (why / how / describe).

Reply with only a JSON object: {{"question": "...", "answer": "...", "evidence": "...", "qtype": "fact|list|explanation"}}"""


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_reply(text: str) -> dict | None:
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None
    return d if all(isinstance(d.get(k), str) and d[k].strip() for k in ("question", "answer", "evidence", "qtype")) else None


def evidence_is_verbatim(evidence: str, passage: str) -> bool:
    return len(evidence.split()) >= 5 and _squash(evidence) in _squash(passage)


def sample_chunks(chunks, n: int, rng: random.Random):
    pool = [c for c in chunks if c.kind == "text" and 150 <= c.n_words <= 420 and not c.section.startswith(SKIP_SECTIONS)]
    rng.shuffle(pool)
    picked, per_sec = [], {}
    for c in pool:
        if per_sec.get(c.section, 0) >= 4:
            continue
        per_sec[c.section] = per_sec.get(c.section, 0) + 1
        picked.append(c)
        if len(picked) == n:
            break
    return picked


def main():
    cfg = yaml.safe_load((ROOT / "config" / "companies.yaml").read_text())
    man = {m["ticker"]: m for m in json.loads((ROOT / "data" / "raw" / "manifest.json").read_text())}
    rng = random.Random(0)
    jobs = []
    for c in cfg["companies"]:
        t = c["ticker"]
        ch = chunk_blocks(blocks_from_html((ROOT / "data" / "raw" / f"{t}_10k.htm").read_bytes()), t, c["short"],
                          ChunkCfg(size=400, overlap=50, header=False))
        jobs += [(c, man[t]["report_date"], x) for x in sample_chunks(ch, PER_COMPANY, rng)]

    def run(job):
        c, end, x = job
        try:
            r = llm.claude(PROMPT.format(name=c["short"], end=end, section=x.section, passage=x.body), model=MODEL, max_tokens=800)
        except Exception as e:
            return {"ticker": c["ticker"], "chunk_id": x.chunk_id, "error": str(e)[:100]}
        d = parse_reply(r)
        if not d:
            return {"ticker": c["ticker"], "chunk_id": x.chunk_id, "error": "unparseable"}
        ok = evidence_is_verbatim(d["evidence"], x.body)
        return {"id": f"txt-{c['ticker']}-{x.chunk_id.split('-')[-1]}", "type": "text", "ticker": c["ticker"], "section": x.section,
                "question": d["question"].strip(), "answer": d["answer"].strip(), "evidence": _squash(d["evidence"]),
                "qtype": d["qtype"], "verbatim": ok, "source_chunk": x.body}

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(run, jobs))
    good = [r for r in res if r.get("verbatim")]
    out = ROOT / "data" / "questions"
    out.mkdir(parents=True, exist_ok=True)
    (out / "text_candidates.jsonl").write_text("\n".join(json.dumps(r) for r in good) + "\n")
    print(f"{len(jobs)} passages -> {len(good)} verified questions;",
          f"{sum('error' in r for r in res)} errors; {sum(1 for r in res if r.get('verbatim') is False)} non-verbatim")
    print("usage:", llm.USAGE)


if __name__ == "__main__":
    main()
