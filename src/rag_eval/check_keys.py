"""Smoke test: one tiny call to each API. Prints no secrets."""
import numpy as np
from . import llm

for m in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"):
    try:
        print(m, "->", llm.claude("Reply with the single word: ok", model=m, max_tokens=300).strip()[:20])
    except Exception as e:
        print(m, "FAILED:", type(e).__name__, str(e)[:200])
try:
    v = llm.embed(["Costco net sales", "revenue of the warehouse retailer"], task="RETRIEVAL_DOCUMENT")
    a, b = np.array(v[0]), np.array(v[1])
    print("gemini embedding ok, dim", len(a), "cosine", round(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), 3))
except Exception as e:
    print("gemini embedding FAILED:", type(e).__name__, str(e)[:200])
print("usage:", llm.USAGE)
