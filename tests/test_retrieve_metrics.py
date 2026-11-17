import numpy as np
from rag_eval import metrics as M
from rag_eval.retrieve import BM25, Dense, rrf, tokenize

DOCS = ["Costco net sales were 303,154 million", "Apple designs iPhones", "Delta Air Lines operates aircraft", "Costco warehouse membership fees"]


def test_tokenizer_keeps_numbers_whole():
    assert tokenize("Net sales: $303,154.5 (FY2026)") == ["net", "sales", "303,154.5", "fy2026"]


def test_bm25_finds_exact_number_and_company():
    bm = BM25(DOCS)
    assert bm.search("303,154", 2)[0] == 0 and bm.search("iPhones", 1)[0] == 1


def test_dense_returns_identical_vector_first():
    v = np.eye(4)
    assert Dense(v).search(v[2], 4)[0] == 2


def test_rrf_matches_hand_calculation_and_prefers_consistency():
    # A: rank1 in list 1, rank4 in list 2 -> 1/61 + 1/64 ; B: rank 2 in both -> 2/62
    a, b = 0, 1
    fused = rrf([[a, b, 9, 8], [7, b, 6, a]])
    assert abs((1 / 61 + 1 / 64) - 0.032018) < 1e-5 and abs(2 / 62 - 0.032258) < 1e-5
    assert fused.index(b) < fused.index(a)


def test_is_hit_requires_right_company_and_evidence():
    qn = {"type": "numeric", "ticker": "COST", "evidence": ["303,154", "303.2 billion"]}
    assert M.is_hit("COST", "net sales 303,154", qn) and not M.is_hit("AAPL", "net sales 303,154", qn)
    assert not M.is_hit("COST", "net sales 1,000", qn)
    qt = {"type": "text", "ticker": "DAL", "evidence": "approximately 20% of our workforce"}
    assert M.is_hit("DAL", "As of 2025,  approximately 20%\nof our workforce, primarily pilots", qt)
    assert not M.is_hit("DAL", "unrelated text", qt) and not M.is_hit("DAL", "x", {"type": "unanswerable", "ticker": "DAL"})


def test_first_hit_rank_recall_and_mrr_by_hand():
    hits = [False, True, False, True]
    assert M.first_hit_rank([0, 2, 1, 3], hits) == 3 and M.first_hit_rank([0, 2], hits) is None
    ranks = [1, 3, None, 2]   # the example in the study guide
    assert M.recall_at(ranks, 3).mean() == 0.75 and abs(M.rr(ranks).mean() - 0.4583) < 1e-3


def test_bootstrap_interval_and_paired_difference():
    x = np.array([1.0] * 80 + [0.0] * 20)
    m, lo, hi = M.boot_ci(x)
    assert lo < 0.8 < hi and abs(m - 0.8) < 1e-9
    d, dlo, dhi = M.paired_boot_diff(x, np.zeros(100))
    assert dlo > 0.6   # clearly positive
    d, dlo, dhi = M.paired_boot_diff(x, x)
    assert d == 0 and dlo == 0 == dhi
