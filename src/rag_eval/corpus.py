"""Build the chunked corpus for all companies under a chunking configuration."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .chunk import Chunk, ChunkCfg, chunk_blocks
from .parse import blocks_from_html

ROOT = Path(__file__).resolve().parents[2]


def build(cfg: ChunkCfg) -> list[Chunk]:
    companies = yaml.safe_load((ROOT / "config" / "companies.yaml").read_text())["companies"]
    man = {m["ticker"]: m for m in json.loads((ROOT / "data" / "raw" / "manifest.json").read_text())}
    out: list[Chunk] = []
    for c in companies:
        t = c["ticker"]
        blocks = blocks_from_html((ROOT / "data" / "raw" / f"{t}_10k.htm").read_bytes())
        out += chunk_blocks(blocks, t, f"{c['short']} | fiscal year ended {man[t]['report_date']}", cfg)
    return out


def load_questions(kinds=("numeric", "text")) -> list[dict]:
    qs = []
    d = ROOT / "data" / "questions"
    for k in kinds:
        f = d / ("text_candidates.jsonl" if k == "text" else f"{k}.jsonl")
        qs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return qs
