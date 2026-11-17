"""LLM judge for free-text answers, plus its validation against known truth.

The judge decides two things separately:
  correct   - does the answer state the key facts of the reference answer?
  supported - is every claim in the answer backed by the excerpts the system was shown?
"""
from __future__ import annotations

import json
import random
import re

from . import llm

JUDGE_MODEL = "claude-haiku-5-5"
CHECK_MODEL = "claude-sonnet-5-5"   # re-judges a sample, as a second opinion on the judge

PROMPT = """You grade answers from a question-answering system about SEC 10-K filings.

Question: {question}
Reference answer (verified from the filing): {reference}

Excerpts the system was shown:
{excerpts}

System answer:
\"\"\"{answer}\"\"\"

Decide:
- "correct": true if the system answer states the key facts of the reference answer (wording and extra detail do not matter; a wrong number, wrong entity, or a missing key part is incorrect; NOT_FOUND is incorrect).
- "supported": true if every factual claim in the system answer can be found in the excerpts above (NOT_FOUND counts as supported; claims from outside knowledge are not supported).

Reply with only JSON: {{"correct": true/false, "supported": true/false, "reason": "<10 words>"}}"""


def parse_verdict(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(d.get("correct"), bool) and isinstance(d.get("supported"), bool):
        return d
    return None


def judge(question: str, reference: str, excerpts: list[str], answer: str, model: str = JUDGE_MODEL) -> dict | None:
    ex = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(excerpts, start=1)) or "(none)"
    out = llm.claude(PROMPT.format(question=question, reference=reference, excerpts=ex, answer=answer),
                     model=model, max_tokens=1500, effort="low")
    return parse_verdict(out)


def _forms(v: float, unit: str) -> list[tuple[str, str]]:
    """(format spec, kind) pairs for how a value is written: '478,834', '478834', '478.8 billion', '8.98'."""
    if unit == "dollars per share":
        return [(f"{abs(v):.2f}", "eps")]
    return [(f"{abs(v):,.0f}", "comma"), (f"{abs(v):.0f}", "plain"), (f"{abs(v) / 1000:,.1f} billion", "bn")]


def corrupt_answer(answer: str, gold: float, unit: str, factor: float) -> str | None:
    """Replace every printed form of the true value (prose and the ANSWER line) with a value off by `factor`.
    Returns None if the true value is not found, so a probe is never accidentally still correct."""
    wrong = gold * factor
    out = answer
    for (g, kind), (w, _) in zip(_forms(gold, unit), _forms(wrong, unit)):
        out = out.replace(g, w)
    if out == answer or any(g in out for g, _ in _forms(gold, unit)):
        return None
    return out


def agreement(pairs: list[tuple[bool, bool]]) -> dict:
    """Judge vs truth: accuracy, and Cohen's kappa (agreement beyond chance)."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    acc = sum(a == b for a, b in pairs) / n
    pj, pt = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pj * pt + (1 - pj) * (1 - pt)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 1.0
    tp = sum(a and b for a, b in pairs); fp = sum(a and not b for a, b in pairs); fn = sum(b and not a for a, b in pairs)
    return {"n": n, "accuracy": acc, "kappa": kappa,
            "false_accept_rate": fp / max(1, sum(not b for _, b in pairs)),
            "false_reject_rate": fn / max(1, sum(b for _, b in pairs)), "tp": tp}
