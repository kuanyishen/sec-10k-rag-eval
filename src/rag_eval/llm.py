"""Thin wrappers around the Claude and Gemini APIs with token/cost accounting."""
from __future__ import annotations

import os
import time

# $ per million tokens (input, output). Used only for the running-cost estimate.
PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}


class Usage:
    def __init__(self):
        self.calls, self.inp, self.out, self.cost = 0, 0, 0, 0.0

    def add(self, model: str, i: int, o: int):
        self.calls += 1; self.inp += i; self.out += o
        pi, po = PRICE.get(model, (0, 0))
        self.cost += i / 1e6 * pi + o / 1e6 * po

    def __repr__(self):
        return f"{self.calls} calls, {self.inp:,} in / {self.out:,} out tokens, ~${self.cost:.3f}"


USAGE = Usage()


def claude(prompt: str, *, model: str, system: str | None = None, max_tokens: int = 1024, effort: str | None = None) -> str:
    """One Claude call, returns the text. Retries on rate limits / overload; raises on refusal."""
    import anthropic
    client = anthropic.Anthropic()
    kw = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    if system:
        kw["system"] = system
    if effort:
        kw["output_config"] = {"effort": effort}
    for attempt in range(6):
        try:
            r = client.messages.create(**kw)
            break
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError("Claude call failed after retries")
    USAGE.add(model, r.usage.input_tokens, r.usage.output_tokens)
    if r.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {getattr(r, 'stop_details', None)}")
    return "".join(b.text for b in r.content if b.type == "text")


def embed(texts: list[str], *, task: str, model: str = "gemini-embedding-001", dim: int = 768) -> list[list[float]]:
    """Gemini embeddings. task: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    out: list[list[float]] = []
    for i in range(0, len(texts), 50):
        for attempt in range(6):
            try:
                r = client.models.embed_content(model=model, contents=texts[i:i + 50],
                                                config=types.EmbedContentConfig(task_type=task, output_dimensionality=dim))
                break
            except Exception:  # rate limit / transient
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("embedding failed after retries")
        out.extend(e.values for e in r.embeddings)
    return out
