# 10-K RAG evaluation: can you trust an AI assistant's answers about an annual report?

**Write-up:** https://kuanyishen.github.io/sec-10k-rag-eval/

A retrieval-augmented question-answering system over the latest 10-K filings of ten companies (Apple, Microsoft, JPMorgan Chase, Walmart, Pfizer, ExxonMobil, Tesla, Costco, Nike, Delta), and an evaluation harness that measures it with checkable ground truth.

## Findings

- **Best setup (dense retrieval, top 5, Claude Haiku 5.5):** 93.5% of numeric answers exactly right (95.4% after correcting 2 grader errors), 97.5% of text answers correct, 30/30 unanswerable questions correctly declined.
- **Retrieval decides answer quality.** Embedding search beat BM25 by 30 points of recall@5. "Hybrid + reranker" was *worse* than embeddings alone here. A one-line company/year header on each chunk added 16 points.
- **The remaining numeric errors are definitions, not hallucinations:** net income with or without minority interests, cash with or without restricted cash, which revenue line.
- **The LLM judge was tested before use:** 95.6% agreement with exact-match truth (kappa 0.91), 60/60 planted errors rejected, 60/60 agreement with a stronger model.

## Evaluation design

| Question set | n | Ground truth | Grading |
|---|---|---|---|
| Numeric | 153 | SEC XBRL facts for the same filing (current and prior year) | Exact match within 0.5% |
| Text | 160 | LLM-written from sampled passages, with a verbatim evidence quote (checked) | Validated LLM judge |
| Unanswerable | 30 | Periods the filing does not report | Must answer NOT_FOUND |

Retrieval is scored separately (recall@k, MRR; a chunk counts if it is from the right company and contains the answer), with bootstrap intervals and paired comparisons.

## Layout

| Path | What |
|---|---|
| `src/rag_eval/edgar.py` | Download 10-Ks and XBRL facts (needs `SEC_USER_AGENT`) |
| `src/rag_eval/parse.py`, `chunk.py` | HTML to paragraphs/tables with Item sections; chunking with size/overlap/header settings |
| `src/rag_eval/questions.py`, `gen_text_questions.py` | Question sets |
| `src/rag_eval/retrieve.py`, `run_retrieval.py` | BM25, Gemini embeddings, RRF hybrid, cross-encoder reranker; retrieval experiment |
| `src/rag_eval/generate.py`, `run_generation.py` | Answering with citations or NOT_FOUND; programmatic grading |
| `src/rag_eval/judge.py`, `run_judge.py` | LLM judge and its validation |
| `analysis/` | Figures, failure labels, page builder |
| `.github/workflows/` | `collect`, `gen-questions`, `retrieval`, `answer` (manual), `ci` |

## Run it

Repository secrets: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `SEC_USER_AGENT` (your name and a contact email; the SEC requires it). Then run the workflows in order: `collect`, `gen-questions`, `retrieval`, `answer`. Locally:

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest -q
PYTHONPATH=src python analysis/retrieval_figs.py && PYTHONPATH=src python analysis/build_report.py
```

Total API cost for the full study was about $2.50 of Claude calls plus Gemini embeddings.

## Limitations

Single-passage questions with templated numeric wording; no multi-year or calculation questions; lenient numeric hit and 0.5% tolerance; arms for answering were chosen after the retrieval results; the judge is validated against programmatic truth, planted errors and a stronger model, not people.
