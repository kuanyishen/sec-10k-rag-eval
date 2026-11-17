"""Retrieval metrics, hit definition, and bootstrap intervals."""
from __future__ import annotations

import re

import numpy as np

KS = (1, 3, 5, 10, 20)


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def is_hit(chunk_ticker: str, chunk_body: str, q: dict) -> bool:
    """A chunk is relevant if it is from the right company AND contains the answer-bearing text:
    any accepted number string (numeric questions) or the verbatim evidence sentence (text questions)."""
    if chunk_ticker != q["ticker"]:
        return False
    if q["type"] == "numeric":
        return any(e in chunk_body for e in q["evidence"])
    if q["type"] == "text":
        return _squash(q["evidence"]) in _squash(chunk_body)
    return False


def first_hit_rank(ranked: list[int], hits: list[bool], limit: int = 20) -> int | None:
    """1-based rank of the first relevant chunk within the top `limit`; None if absent."""
    for r, idx in enumerate(ranked[:limit], start=1):
        if hits[idx]:
            return r
    return None


def recall_at(ranks: list, k: int) -> np.ndarray:
    return np.array([1.0 if (r is not None and r <= k) else 0.0 for r in ranks])


def rr(ranks: list) -> np.ndarray:
    return np.array([1.0 / r if r is not None else 0.0 for r in ranks])


def boot_ci(x: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile interval over resampled questions."""
    rng = np.random.default_rng(seed)
    n = len(x)
    means = x[rng.integers(0, n, (reps, n))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_boot_diff(a: np.ndarray, b: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) on the same questions, with a 95% interval. Interval excluding 0 = a real difference."""
    return boot_ci(a - b, reps, seed)
