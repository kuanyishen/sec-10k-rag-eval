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


def corrupt_number(answer: str, rng: random.Random) -> str | None:
    """Change the first multi-digit number in an answer by 10-30%, keeping its format. Used to make known-wrong answers."""
    m = re.search(r"\d[\d,]*\.?\d*", answer)
    if not m or len(re.sub(r"\D", "", m.group(0))) < 2:
        return None
    raw = m.group(0)
    v = float(raw.replace(",", ""))
    nv = v * rng.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
    dec = len(raw.split(".")[1]) if "." in raw else 0
    new = f"{nv:,.{dec}f}" if "," in raw else f"{nv:.{dec}f}"
    return answer[: m.start()] + new + answer[m.end():]


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
