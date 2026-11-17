"""Build docs/index.html and the answer-level figure from the committed results (no API calls)."""
import glob, json, sys
from collections import Counter
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from rag_eval import metrics as M

A = ROOT / "docs" / "assets"
REPO = "https://github.com/kuanyishen/sec-10k-rag-eval"
RT = json.loads((ROOT / "results" / "retrieval_tables.json").read_text())
JV = json.loads((ROOT / "results" / "judge_validation.json").read_text())
FL = {k: v for k, v in json.loads((ROOT / "analysis" / "failure_labels.json").read_text()).items() if not k.startswith("_")}
MAN = json.loads((ROOT / "data" / "raw" / "manifest.json").read_text())
ROWS = [json.loads(l) for f in sorted(glob.glob(str(ROOT / "results" / "generation" / "*.jsonl"))) for l in open(f)]
VER = {}
for l in open(ROOT / "results" / "judge" / "verdicts.jsonl"):
    x = json.loads(l)
    if x["tag"] == "text" and x["verdict"]:
        VER[(x["id"], x["arm"])] = x["verdict"]
ARMS = {"closed_book": "No retrieval (model memory)", "bm25_k5": "BM25, top 5", "hybrid_k5": "Hybrid, top 5",
        "hybrid_rerank_k5": "Hybrid + reranker, top 5", "dense_k3": "Dense, top 3", "dense_k5": "Dense, top 5",
        "dense_k10": "Dense, top 10", "oracle_k5": "Dense, answer passage guaranteed"}
pc = lambda x, d=0: f"{x * 100:.{d}f}%"


def arm_stats(a):
    R = [r for r in ROWS if r["arm"] == a]
    num = np.array([r["correct"] for r in R if r["type"] == "numeric"], float)
    tv = [VER.get((r["id"], a)) for r in R if r["type"] == "text"]
    tc = np.array([v["correct"] for v in tv if v], float)
    ts = np.array([v["supported"] for v in tv if v], float)
    ans = [r for r in R if r["type"] != "unanswerable"]
    una = [r for r in R if r["type"] == "unanswerable"]
    return {"num": M.boot_ci(num), "n_num": len(num), "text": M.boot_ci(tc), "n_text": len(tc), "supported": float(ts.mean()),
            "false_abstain": float(np.mean([r["abstained"] for r in ans])), "una_abstain": float(np.mean([r["abstained"] for r in una])),
            "una_n": len(una), "num_k": int(num.sum()), "text_k": int(tc.sum()), "una_k": int(sum(r["abstained"] for r in una))}


S = {a: arm_stats(a) for a in ARMS}

# figure: answer correctness by arm
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
fig, ax = plt.subplots(figsize=(7.6, 4.8))
names = list(ARMS)
y = np.arange(len(names)); h = 0.38
for j, (key, col, lab) in enumerate((("num", "#0072B2", "numeric (exact match)"), ("text", "#D55E00", "text (validated judge)"))):
    v = [S[a][key] for a in names]
    ax.barh(y + (j - 0.5) * h, [x[0] for x in v], h, color=col, label=lab,
            xerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]], capsize=2.5)
ax.set_yticks(y, [ARMS[a] for a in names]); ax.invert_yaxis(); ax.set_xlim(0, 1.0)
ax.set_xlabel("Share of answers correct (95% interval)")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
ax.set_title("Answer accuracy by retrieval setup", loc="left", fontsize=12, pad=26)
fig.tight_layout(); fig.savefig(A / "fig_answers.png"); plt.close(fig)

# --- tables -------------------------------------------------------------------------------
def ci(t, d=0):
    return f"{pc(t[0], d)} <span class='sub'>({pc(t[1], d)}–{pc(t[2], d)})</span>"


t_ans = ("<div class='tablewrap'><table><tr><th>Setup</th><th>Numeric correct</th><th>Text correct</th><th>Text supported by excerpts</th>"
         "<th>Answerable but said NOT_FOUND</th><th>Unanswerable: abstained</th></tr>"
         + "".join(f"<tr><td>{ARMS[a]}</td><td>{ci(S[a]['num'])}</td><td>{ci(S[a]['text'])}</td><td>{pc(S[a]['supported'])}</td>"
                   f"<td>{pc(S[a]['false_abstain'], 1)}</td><td>{S[a]['una_k']}/{S[a]['una_n']}</td></tr>" for a in ARMS)
         + "</table></div>")
RN = {"bm25": "BM25 (keywords)", "dense": "Dense (embeddings)", "hybrid": "Hybrid (RRF)", "dense_rerank": "Dense + reranker",
      "hybrid_rerank": "Hybrid + reranker", "hybrid_oracle_company": "Hybrid, company known in advance"}
bt = RT["by_type"]
t_ret = ("<div class='tablewrap'><table><tr><th>Retriever (400-word chunks, header)</th><th>Recall@5, all</th><th>Numeric</th><th>Text</th><th>MRR, all</th></tr>"
         + "".join(f"<tr><td>{RN[r]}</td><td>{ci(bt[r]['all']['recall@5'])}</td><td>{pc(bt[r]['numeric']['recall@5'][0])}</td>"
                   f"<td>{pc(bt[r]['text']['recall@5'][0])}</td><td>{bt[r]['all']['mrr'][0]:.2f}</td></tr>" for r in RN)
         + "</table></div>")
D = RT["diffs"]
dfmt = lambda k, scale=100: f"{D[k][0] * scale:+.1f} points (95% CI {D[k][1] * scale:+.1f} to {D[k][2] * scale:+.1f})"
fl = Counter(v[0] for k, v in FL.items() if k.startswith("num-"))
jn, js = JV["numeric_correct_vs_truth"], JV["second_opinion_correct_agreement"]
cr = JV["corrupted_answers_rejected"]; cr_old = JV.get("corrupted_answers_rejected_v1_buggy_probe") or {}
d5 = S["dense_k5"]
n_tests = sum(1 for f in (ROOT / "tests").glob("test_*.py") for l in f.read_text().splitlines() if l.startswith("def test_"))
fy26 = sum(1 for m in MAN if m["report_date"] >= "2026-01-01")
cb = [r for r in ROWS if r["arm"] == "closed_book" and r["type"] == "numeric"]
yr = {m["ticker"]: int(m["report_date"][:4]) for m in MAN}
qmeta = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data" / "questions" / "numeric.jsonl")}
def cb_year(y):
    s = [r["correct"] for r in cb if yr[r["ticker"]] - (qmeta[r["id"]]["period"] == "prior") == y]
    return sum(s), len(s)
c26, n26 = cb_year(2026)

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>10-K RAG Evaluation</title>
<meta name="description" content="A retrieval-augmented question-answering system over ten SEC 10-K filings, and an evaluation harness that measures retrieval, answer correctness, faithfulness and abstention with checkable ground truth.">
<link rel="stylesheet" href="style.css"></head>
<body>
<nav class="top"><div class="wrap"><a class="brand" href="index.html">10-K RAG evaluation</a>
<a href="#design">Evaluation design</a><a href="#retrieval">Retrieval</a><a href="#answers">Answers</a><a href="#judge">Judge</a><a href="#failures">Failures</a><a href="#limits">Limitations</a><a href="{REPO}">GitHub</a></div></nav>
<main class="wrap">
<header class="hero">
<div class="eyebrow">RAG · evaluation · SEC EDGAR · Claude · Gemini embeddings</div>
<h1>Can you trust an AI assistant's answers about a company's annual report, and how would you know?</h1>
<p class="lead">I built a question-answering system over ten companies' latest 10-K filings, but most of the work is the evaluation harness: 343 questions with checkable answers, retrieval and answer quality measured separately, and an LLM judge that is itself tested before its scores are used.</p>
<div class="row"><a class="btn" href="#answers">See the results</a><a class="btn ghost" href="{REPO}">View code</a></div></header>

<div class="kpis">
<div class="kpi"><b>{pc(d5['num'][0], 1)}</b><span>numeric answers exactly right with the best setup ({d5['num_k']}/{d5['n_num']}); {fl['grader']} more were grader errors</span></div>
<div class="kpi"><b>{pc(d5['text'][0], 1)}</b><span>free-text answers judged correct ({d5['text_k']}/{d5['n_text']})</span></div>
<div class="kpi"><b>{S['dense_k5']['una_k']}/{S['dense_k5']['una_n']}</b><span>unanswerable questions where the system said "not found" instead of guessing</span></div>
<div class="kpi"><b>+{D['header_minus_noheader_dense'][0] * 100:.0f} pts</b><span>retrieval gain from one design choice: labelling every chunk with its company and year</span></div></div>

<div class="callout"><p><b>TL;DR.</b> With good retrieval, a small, cheap model (Claude Haiku 5.5) answers questions about these filings very accurately and almost never invents an answer. The decisive choices were in retrieval, not generation: embedding search beat keyword search by {D['dense_minus_bm25'][0] * 100:.0f} points, labelling each chunk with its company added {D['header_minus_noheader_dense'][0] * 100:.0f}, and the popular "hybrid + reranker" recipe was <i>worse</i> than plain embedding search here. Most remaining "errors" are not hallucinations but definitions: net income with or without minority interests, cash with or without restricted cash. A real deployment would need the question to say which one it means.</p></div>

<h2 id="question">The business question</h2>
<p>Analysts, auditors and investors spend hours finding facts in 100-to-300-page annual reports. An AI assistant that answers from the filing could save that time, but only if you know how often it is right, and whether it says "I don't know" instead of making something up. So the question is not "can I build a chatbot" but <b>how accurate is it, which design choices matter, and where does it fail?</b></p>

<h2 id="data">The data</h2>
<p>The latest annual report (10-K) of ten large companies across sectors: Apple, Microsoft, JPMorgan Chase, Walmart, Pfizer, ExxonMobil, Tesla, Costco, Nike and Delta. They were downloaded from SEC EDGAR by a GitHub Actions job, together with the SEC's machine-readable XBRL facts for the same filings. {fy26} of the ten are fiscal-2026 reports; without retrieval the model got {c26} of {n26} of their figures right, so for those it has to use the documents. The parser turns each filing's HTML into paragraphs and tables, tags each with its Item section (Risk Factors, MD&amp;A, and so on), and splits them into about 4,600 chunks of roughly 400 words.</p>

<h2 id="design">Evaluation design</h2>
<p>The design keeps two questions apart: <b>did retrieval find the right passage?</b> and <b>given the passages, did the model answer correctly and faithfully?</b></p>
<ul class="tight">
<li><b>153 numeric questions</b> ("What was Costco's net income for the fiscal year ended August 30, 2026?"), with answers taken from the SEC's XBRL facts for that exact filing, for the current and prior year. Graded by exact match within rounding, with no LLM involved.</li>
<li><b>160 text questions</b> written by Claude Sonnet 5.5 from randomly sampled passages, each with a quote from the filing that contains the answer. Any question whose quote was not found word for word in the passage was discarded (none were).</li>
<li><b>30 unanswerable questions</b> about years the filing does not cover (five years earlier, or the next fiscal year). The right behaviour is to say NOT_FOUND.</li>
<li><b>Retrieval hit:</b> a chunk counts if it is from the right company and contains the answer (the number, or the verbatim quote). Metrics: recall@5 and MRR, with bootstrap intervals over questions and paired comparisons on the same questions.</li>
</ul>

<h2 id="retrieval">Retrieval: what finds the answer?</h2>
{t_ret}
<figure><img src="assets/fig_retrievers.png" alt="Recall by retriever and question type"><figcaption>Error bars are 95% bootstrap intervals over questions.</figcaption></figure>
<ul class="tight">
<li><b>Embedding (dense) search beats keyword search (BM25)</b> by {dfmt('dense_minus_bm25')}. The gap is almost entirely on numeric questions: BM25 matches generic words such as "net income" and "fiscal year" in hundreds of chunks.</li>
<li><b>Hybrid search was worse than dense alone</b>, by {dfmt('dense_minus_hybrid')}. Merging rankings let BM25's wrong chunks displace correct ones on numeric questions. On text questions hybrid was slightly ahead ({dfmt('text_hybrid_minus_dense')}), which is within noise.</li>
<li><b>The off-the-shelf reranker hurt</b>: {dfmt('dense_minus_dense_rerank')} on recall, and {D['dense_minus_dense_rerank_mrr'][0]:+.2f} on MRR. It was trained on web search queries and cuts long chunks to 512 tokens.</li>
<li><b>Knowing the company in advance</b> (filtering to the right filing) helped hybrid only a little, so cross-company confusion was not the main problem once chunks carry a header.</li>
</ul>
<h3>Chunking choices</h3>
<figure><img src="assets/fig_chunking.png" alt="Chunking choices" style="max-width:620px"><figcaption>Dense retrieval, all 313 answerable questions.</figcaption></figure>
<p>The biggest single lever was a one-line <b>header on every chunk</b> ("Costco 10-K, fiscal year ended August 30, 2026 | Item 7 MD&amp;A"): {dfmt('header_minus_noheader_dense')}. Without it, a chunk of a financial table does not say whose table it is. 400-word chunks beat 150-word chunks ({dfmt('s400_minus_s150_dense')}), partly because small chunks split 5% of answers across two pieces. 400 vs 800 words was within noise ({dfmt('s400_minus_s800_dense')}).</p>
<p class="sub">Note: my first header wrote dates as 2026-08-30 while questions say "August 30, 2026". I suspected that handicapped BM25 and changed it, but BM25 recall barely moved (0.693 to 0.684), so date format was not the explanation.</p>

<h2 id="answers">Answers: correct, faithful, and honest about gaps</h2>
<p>Claude Haiku 5.5 answered every question under eight setups, using only the excerpts given and citing them, or replying NOT_FOUND. I chose these setups after seeing the retrieval results, so they lean towards dense retrieval.</p>
{t_ans}
<figure><img src="assets/fig_answers.png" alt="Answer accuracy by setup"></figure>
<ul class="tight">
<li><b>Retrieval quality drives answer quality.</b> Numeric accuracy goes from {pc(S['bm25_k5']['num'][0])} with BM25 to {pc(S['dense_k5']['num'][0])} with dense retrieval, using the same model and prompt.</li>
<li><b>When retrieval fails, the model mostly says so.</b> With BM25, {pc(S['bm25_k5']['false_abstain'])} of answerable questions got NOT_FOUND rather than a guessed number.</li>
<li><b>No retrieval means no reliable answers.</b> From memory alone the model got {pc(S['closed_book']['num'][0])} of numeric questions right, almost all for older years: {c26}/{n26} for fiscal-2026 figures it could not have seen.</li>
<li><b>Abstention:</b> with any retrieval, it declined all 30 unanswerable questions. Without retrieval it answered {S['closed_book']['una_n'] - S['closed_book']['una_k']} of them, often with real figures from memory; they are unanswerable only from these filings.</li>
<li><b>Guaranteeing the answer passage did not raise numeric accuracy</b> ({pc(S['oracle_k5']['num'][0], 1)}, same as dense top 5). The remaining numeric errors are not retrieval failures.</li>
</ul>

<h2 id="judge">Can the LLM judge be trusted?</h2>
<p>Text answers are graded by an LLM judge (Claude Haiku 5.5), so I tested the judge before using its scores:</p>
<ul class="tight">
<li><b>Against known truth:</b> on {jn['n']} numeric answers where exact-match grading gives the truth, the judge agreed {pc(jn['accuracy'], 1)} of the time (Cohen's kappa {jn['kappa']:.2f}). It never accepted a wrong answer. Where they disagreed ({pc(jn['false_reject_rate'])} of exact-match-correct answers), the judge was <i>stricter</i> than my rounding rule: it rejected rounded figures ("$9.0 billion" for $9,037M) and two JPMorgan totals the model had estimated from a percentage, which the 0.5% tolerance accepts. That is leniency in my exact-match grader, not judge error.</li>
<li><b>Against planted errors:</b> I took correct answers and changed the number by 10–30%. The judge rejected {cr['rejected']}/{cr['n']}.</li>
<li><b>Against a stronger model:</b> Claude Sonnet 5.5 re-graded {js['n']} text answers and agreed on all {js['n']}.</li>
</ul>
<div class="callout warn"><p><b>A mistake I caught in my own harness.</b> The first version of the planted-error test changed the <i>first</i> number in each answer, which was often a date ("September 26" to "September 29") and left the final answer untouched. The judge "rejected" only {cr_old.get('rejected', '?')}/{cr_old.get('n', '?')}, which looked like a bad judge. After fixing the test to change the answer itself, it rejected {cr['rejected']}/{cr['n']}. Testing the tester matters.</p></div>
<p class="sub">Limit: a "supported by excerpts" check against the numeric ground truth was uninformative, because nearly every answer was supported (kappa about 0 on a near-constant label).</p>

<h2 id="failures">What the remaining errors are</h2>
<p>I read every miss of the best setup (dense, top 5):</p>
<ul class="tight">
<li><b>{fl['definition']} of {d5['n_num'] - d5['num_k']} numeric misses are definitional.</b> The model chose net income including minority interests (Walmart, Tesla), equity including minority interests (Walmart), cash including restricted cash (Exxon) or a different revenue line (Exxon). In most of them the answer also stated the figure the SEC data uses. The question did not say which definition it meant, and in practice an analyst would specify.</li>
<li><b>{fl['grader']} were grader errors.</b> The SEC's structured data rounds Exxon's R&amp;D spending to $1.2B and $1.0B, while the filing's table and the model say $1,228M and $987M. Counting those as correct, numeric accuracy is {(d5['num_k'] + fl['grader']) / d5['n_num'] * 100:.1f}%.</li>
<li><b>All {d5['n_text'] - d5['text_k']} text misses were retrieval misses.</b> In 3 the model said NOT_FOUND. In 1 it answered with a related, broader figure and said the exact one was not in the excerpts.</li>
<li><b>No miss was a fabricated number.</b></li>
</ul>

<h2 id="build">How it is built</h2>
<ul class="tight">
<li><b>Pipeline:</b> EDGAR download and XBRL facts → block-aware HTML parser (tables kept as rows, Item sections detected, including filers who put headings in one-row tables) → chunker (size, overlap and header as settings) → BM25 / Gemini embeddings / RRF / cross-encoder.</li>
<li><b>Runs on GitHub Actions</b> with API keys as repository secrets; the development sandbox could not reach the SEC.</li>
<li><b>{n_tests} unit tests</b>, including hand-computed RRF, recall and MRR values, the parser on tricky HTML, verbatim-evidence checks, and the corruption probe.</li>
<li><b>Cost:</b> about $2.50 of Claude calls in total (question writing, 2,744 answers, about 1,600 judge calls), plus Gemini embeddings.</li>
</ul>

<h2 id="limits">Limitations</h2>
<ul class="tight">
<li><b>The questions are easier than real analyst questions.</b> Each needs one passage, and numeric ones follow fixed templates that dense search suits well. Multi-year comparisons, calculations and multi-hop questions are not tested.</li>
<li><b>Generated text questions</b> come from passages the parser kept, so they cannot test text the parser lost (for example, content in images).</li>
<li><b>Lenient retrieval hit for numbers:</b> any chunk from the right company containing the figure counts, even a passing mention. <b>Lenient answer grading:</b> the 0.5% tolerance accepts rounded and estimated figures.</li>
<li><b>Setups were chosen after seeing retrieval results</b> on the same questions, and I compared many variants, so small differences should not be over-read.</li>
<li><b>No human-labelled set.</b> The judge was validated against programmatic truth, planted errors and a stronger model, not against people.</li>
<li><b>One model per role, ten companies, one filing each.</b></li>
</ul>
<footer>Filings: SEC EDGAR (public). Code and tests: <a href="{REPO}">{REPO.split('github.com/')[1]}</a>.</footer>
</main></body></html>"""
(ROOT / "docs" / "index.html").write_text(page)
print("wrote docs/index.html")
