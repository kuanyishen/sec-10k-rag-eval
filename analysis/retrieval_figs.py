"""Figures and tables for the retrieval experiment (no API calls)."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from rag_eval import metrics as M
from rag_eval.corpus import load_questions

A = ROOT / "docs" / "assets"; A.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
QS = {q["id"]: q for q in load_questions()}
NAMES = {"bm25": "BM25 (keywords)", "dense": "Dense (embeddings)", "hybrid": "Hybrid (RRF)", "dense_rerank": "Dense + reranker",
         "hybrid_rerank": "Hybrid + reranker", "hybrid_oracle_company": "Hybrid, company known"}
CFGS = {"s400_o50_h": "400 words, header", "s400_o50_nh": "400 words, no header", "s150_o25_h": "150 words, header", "s800_o100_h": "800 words, header"}


def ranks(cfg, r):
    return {x["id"]: x["rank"] for x in json.loads((ROOT / "results" / "retrieval" / f"{cfg}__{r}.json").read_text())}


def main():
    out = {"by_type": {}, "configs": {}, "diffs": {}}
    cfg = "s400_o50_h"
    R = {r: ranks(cfg, r) for r in NAMES}
    ids = list(R["dense"])
    for r in NAMES:
        row = {}
        for t in ("all", "numeric", "text"):
            sel = [i for i in ids if t == "all" or QS[i]["type"] == t]
            row[t] = {"recall@5": M.boot_ci(M.recall_at([R[r][i] for i in sel], 5)), "mrr": M.boot_ci(M.rr([R[r][i] for i in sel]))}
        out["by_type"][r] = row
    for c in CFGS:
        rk = ranks(c, "dense")
        out["configs"][c] = {"recall@5": M.boot_ci(M.recall_at([rk[i] for i in ids], 5)), "mrr": M.boot_ci(M.rr([rk[i] for i in ids]))}

    def diff(a, b, ca=cfg, cb=cfg, metric="recall"):
        A_, B_ = ranks(ca, a), ranks(cb, b)
        f = (lambda x: M.recall_at(x, 5)) if metric == "recall" else M.rr
        return M.paired_boot_diff(f([A_[i] for i in ids]), f([B_[i] for i in ids]))
    out["diffs"] = {
        "dense_minus_bm25": diff("dense", "bm25"), "dense_minus_hybrid": diff("dense", "hybrid"),
        "dense_minus_dense_rerank": diff("dense", "dense_rerank"), "dense_minus_dense_rerank_mrr": diff("dense", "dense_rerank", metric="mrr"),
        "header_minus_noheader_dense": diff("dense", "dense", cfg, "s400_o50_nh"),
        "s400_minus_s150_dense": diff("dense", "dense", cfg, "s150_o25_h"), "s400_minus_s800_dense": diff("dense", "dense", cfg, "s800_o100_h"),
    }
    # text-only paired: hybrid vs dense
    tsel = [i for i in ids if QS[i]["type"] == "text"]
    out["diffs"]["text_hybrid_minus_dense"] = M.paired_boot_diff(M.recall_at([R["hybrid"][i] for i in tsel], 5), M.recall_at([R["dense"][i] for i in tsel], 5))
    (ROOT / "results" / "retrieval_tables.json").write_text(json.dumps(out, indent=1))

    # figure: recall@5 by retriever and question type
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    names = ["bm25", "dense", "hybrid", "dense_rerank", "hybrid_rerank"]
    x = np.arange(len(names)); w = 0.38
    for j, (t, col) in enumerate((("numeric", "#0072B2"), ("text", "#D55E00"))):
        m = [out["by_type"][n][t]["recall@5"] for n in names]
        ax.bar(x + (j - 0.5) * w, [v[0] for v in m], w, color=col, label=f"{t} questions",
               yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]], capsize=3)
    ax.set_xticks(x, [NAMES[n].replace(" (", "\n(") for n in names], fontsize=9.5)
    ax.set_ylim(0, 1.18); ax.set_ylabel("Recall@5 (answer in top 5)"); ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title("Which retriever finds the answer? Depends on the question", loc="left", fontsize=12)
    fig.tight_layout(); fig.savefig(A / "fig_retrievers.png"); plt.close(fig)

    # figure: chunking / header
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cs = list(CFGS)
    v = [out["configs"][c]["recall@5"] for c in cs]
    ax.barh([CFGS[c] for c in cs][::-1], [a[0] for a in v][::-1], color=["#0072B2" if c != "s400_o50_nh" else "#D55E00" for c in cs][::-1],
            xerr=[[a[0] - a[1] for a in v][::-1], [a[2] - a[0] for a in v][::-1]], capsize=3)
    ax.set_xlim(0.6, 1.0); ax.set_xlabel("Dense retrieval recall@5, all 313 questions")
    ax.set_title("Chunking choices (dense retrieval)", loc="left", fontsize=12)
    fig.tight_layout(); fig.savefig(A / "fig_chunking.png"); plt.close(fig)
    print(json.dumps(out["diffs"], indent=1))


if __name__ == "__main__":
    main()
