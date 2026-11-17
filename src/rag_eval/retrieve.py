"""Retrievers: BM25 (keywords), dense (embeddings), hybrid (reciprocal rank fusion), cross-encoder rerank."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(s: str) -> list[str]:
    """Lowercase words and numbers; '1,234.5' stays one token so exact figures can match."""
    return TOKEN.findall(s.lower())


class BM25:
    def __init__(self, texts: list[str]):
        self.bm = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int = 50, allowed: np.ndarray | None = None) -> list[int]:
        s = self.bm.get_scores(tokenize(query))
        if allowed is not None:
            s = np.where(allowed, s, -np.inf)
        return np.argsort(-s, kind="stable")[:k].tolist()


class Dense:
    def __init__(self, vectors: np.ndarray):
        v = np.asarray(vectors, dtype=np.float32)
        self.v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def search(self, qvec, k: int = 50, allowed: np.ndarray | None = None) -> list[int]:
        q = np.asarray(qvec, dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-9)
        s = self.v @ q
        if allowed is not None:
            s = np.where(allowed, s, -np.inf)
        return np.argsort(-s, kind="stable")[:k].tolist()


def rrf(rankings: list[list[int]], k: int = 60, top: int = 50) -> list[int]:
    """Reciprocal rank fusion: each list gives a chunk 1/(k+rank) points; sum and sort."""
    score: dict[int, float] = {}
    for r in rankings:
        for rank, idx in enumerate(r, start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank)
    return [i for i, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))][:top]


class Reranker:
    def __init__(self, model: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(model, max_length=512)

    def rerank(self, query: str, texts: list[str], idxs: list[int], top: int = 20) -> list[int]:
        scores = self.m.predict([(query, texts[i]) for i in idxs], batch_size=32, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [idxs[i] for i in order][:top]
