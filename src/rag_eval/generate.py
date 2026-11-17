"""Answer step: prompt building, answer parsing, and exact-match grading for numeric questions."""
from __future__ import annotations

import re

SYSTEM = """You answer questions about SEC 10-K annual reports using ONLY the numbered excerpts provided.
- Cite the excerpts you used like [2].
- Do not use outside knowledge. If the excerpts do not contain the answer, reply with exactly: NOT_FOUND
- Pay attention to the company and the fiscal period in the question; excerpts may come from other years or companies.
Be concise (one to three sentences)."""

SYSTEM_CLOSED_BOOK = """Answer the question about a company's SEC 10-K annual report from your own knowledge.
If you do not know or are not sure, reply with exactly: NOT_FOUND
Be concise (one to three sentences)."""

NUM_SUFFIX = "\nThis is a numeric question: end with a final line `ANSWER: <number>` giving the value in {unit} as a plain number (no units or commas)."


def build_prompt(question: dict, excerpts: list[str] | None) -> tuple[str, str]:
    """-> (system, user). excerpts=None means closed-book (no retrieval)."""
    suffix = NUM_SUFFIX.format(unit=question["unit"]) if question["type"] == "numeric" else ""
    if excerpts is None:
        return SYSTEM_CLOSED_BOOK, f"Question: {question['question']}{suffix}"
    ex = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(excerpts, start=1))
    return SYSTEM, f"Excerpts:\n{ex}\n\nQuestion: {question['question']}{suffix}"


def abstained(answer: str) -> bool:
    return "NOT_FOUND" in answer.upper().replace(" ", "_")


_NUM = re.compile(r"ANSWER:\s*\$?\s*(\(?-?[\d,]*\.?\d+\)?)", re.I)


def parse_number(answer: str) -> float | None:
    m = _NUM.search(answer)
    if not m:
        return None
    s = m.group(1)
    neg = s.startswith("(") or s.startswith("-")
    try:
        v = float(re.sub(r"[(),\-]", "", s))
    except ValueError:
        return None
    return -v if neg else v


def numeric_correct(answer: str, gold: float, unit: str) -> bool:
    """Exact-match grading with rounding tolerance: 0.5% relative (or 0.005 for per-share values).
    Signs are ignored because models often write a loss as a positive 'net loss of X'."""
    v = parse_number(answer)
    if v is None:
        return False
    if unit == "dollars per share":
        return abs(abs(v) - abs(gold)) <= 0.005
    return abs(abs(v) - abs(gold)) <= 0.005 * abs(gold)


def number_in_context(question: dict, excerpts: list[str]) -> bool:
    """Programmatic faithfulness for numeric questions: is a printed form of the true value in what the model saw?"""
    return any(e in t for t in excerpts for e in question["evidence"])
