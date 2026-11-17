"""Retrieval experiment: every retriever x every chunking configuration, scored against the answer-bearing chunks.

    python -m rag_eval.run_retrieval              (needs GEMINI_API_KEY; runs on GitHub Actions)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import llm, metrics as M
from .chunk import ChunkCfg
from .corpus import build, load_questions
from .retrieve import BM25, Dense, Reranker, rrf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "retrieval"
CONFIGS = [ChunkCfg(400, 50, True), ChunkCfg(400, 50, False), ChunkCfg(150, 25, True), ChunkCfg(800, 100, True)]
RETRIEVERS = ["bm25", "dense", "hybrid", "dense_rerank", "hybrid_rerank", "hybrid_oracle_company"]
POOL = 30        # candidates handed to the reranker
EMBED_CHARS = 5500


def cached_embeddings(name: str, texts: list[str]) -> np.ndarray:
    """Embeddings are deterministic for a given text, so keep them between runs (GitHub cache)."""
    key = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
    f = ROOT / "data" / "index" / f"{name}_{key}.npy"
    if f.exists():
        return np.load(f)
    v = np.array(llm.embed(texts, task="RETRIEVAL_DOCUMENT"), dtype=np.float32)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, v)
    return v


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qs = load_questions()
    print(len(qs), "questions")
    qvecs = np.array(llm.embed([q["question"] for q in qs], task="RETRIEVAL_QUERY"))
    print("query embeddings done", flush=True)
    reranker = Reranker()
    summary = {}
    for cfg in CONFIGS:
        chunks = build(cfg)
        texts = [c.text for c in chunks]
        tickers = np.array([c.ticker for c in chunks])
        print(cfg.name, len(chunks), "chunks")
        dense = Dense(cached_embeddings(cfg.name, [t[:EMBED_CHARS] for t in texts]))
        bm = BM25(texts)
        # which chunks hold the answer, per question (independent of the retriever)
        hits = [[M.is_hit(c.ticker, c.body, q) for c in chunks] for q in qs]
        reachable = [any(h) for h in hits]
        runs = {r: [] for r in RETRIEVERS}
        for qi, q in enumerate(qs):
            allowed = tickers == q["ticker"]
            b, d = bm.search(q["question"], 50), dense.search(qvecs[qi], 50)
            h = rrf([b, d])
            rr_ = reranker.rerank(q["question"], texts, h[:POOL], top=20)
            dr = reranker.rerank(q["question"], texts, d[:POOL], top=20)
            ho = rrf([bm.search(q["question"], 50, allowed), dense.search(qvecs[qi], 50, allowed)])
            for name, ranked in (("bm25", b), ("dense", d), ("hybrid", h), ("dense_rerank", dr), ("hybrid_rerank", rr_), ("hybrid_oracle_company", ho)):
                runs[name].append({"id": q["id"], "rank": M.first_hit_rank(ranked, hits[qi]),
                                   "top": [chunks[i].chunk_id for i in ranked[:10]]})
        summary[cfg.name] = {"n_chunks": len(chunks), "reachable": float(np.mean(reachable)), "retrievers": {}}
        for name in RETRIEVERS:
            ranks = [r["rank"] for r in runs[name]]
            row = {"mrr": M.boot_ci(M.rr(ranks))}
            for k in M.KS:
                row[f"recall@{k}"] = M.boot_ci(M.recall_at(ranks, k))
            for t in ("numeric", "text"):
                idx = [i for i, q in enumerate(qs) if q["type"] == t]
                row[f"recall@5_{t}"] = M.boot_ci(M.recall_at([ranks[i] for i in idx], 5))
                row[f"mrr_{t}"] = M.boot_ci(M.rr([ranks[i] for i in idx]))
            summary[cfg.name]["retrievers"][name] = row
            (OUT / f"{cfg.name}__{name}.json").write_text(json.dumps(runs[name]))
            print(f"  {name:24s} recall@5 {row['recall@5'][0]:.3f}  mrr {row['mrr'][0]:.3f}", flush=True)
        (ROOT / "results" / "retrieval_summary.json").write_text(json.dumps(summary, indent=1))  # save as we go
    (ROOT / "results" / "retrieval_summary.json").write_text(json.dumps(summary, indent=1))
    print("usage:", llm.USAGE)


if __name__ == "__main__":
    main()
