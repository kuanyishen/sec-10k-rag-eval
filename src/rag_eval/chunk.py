"""Blocks -> chunks. Three knobs are experiment arms: size, overlap, and the context header."""
from __future__ import annotations

from dataclasses import dataclass

from .parse import Block


@dataclass(frozen=True)
class ChunkCfg:
    size: int = 400        # target words per chunk
    overlap: int = 50      # words repeated from the previous chunk (text only)
    header: bool = True    # prepend "[Company | FY | Section]" so the chunk names its company

    @property
    def name(self) -> str:
        return f"s{self.size}_o{self.overlap}_{'h' if self.header else 'nh'}"


@dataclass
class Chunk:
    chunk_id: str
    ticker: str
    section: str
    kind: str
    body: str          # chunk text without header
    text: str          # what gets indexed (header + body)
    n_words: int


def _words(s: str) -> int:
    return len(s.split())


def _split_long(text: str, size: int) -> list[str]:
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size)]


def _table_chunks(text: str, size: int) -> list[str]:
    if _words(text) <= int(size * 1.5):
        return [text]
    rows = text.split("\n")
    head = rows[:2]
    parts, cur, n = [], [], _words("\n".join(head))
    for r in rows[2:]:
        rw = _words(r)
        if cur and n + rw > size:
            parts.append("\n".join(head + cur)); cur, n = [], _words("\n".join(head))
        cur.append(r); n += rw
    if cur:
        parts.append("\n".join(head + cur))
    return parts


def chunk_blocks(blocks: list[Block], ticker: str, label: str, cfg: ChunkCfg) -> list[Chunk]:
    """`label` is the human context, e.g. 'Costco | fiscal year ended 2026-08-30'."""
    chunks: list[Chunk] = []

    def emit(section: str, kind: str, body: str):
        head = f"[{label} | {section}]\n" if cfg.header else ""
        chunks.append(Chunk(f"{ticker}-{cfg.name}-{len(chunks):05d}", ticker, section, kind, body, head + body, _words(body)))

    cur: list[str] = []
    cur_sec = None

    def flush_text():
        nonlocal cur
        if cur:
            emit(cur_sec, "text", "\n".join(cur))
        cur = []

    for b in blocks:
        if b.section != cur_sec:
            flush_text(); cur_sec = b.section
        if b.kind == "table":
            flush_text()
            for part in _table_chunks(b.text, cfg.size):
                emit(b.section, "table", part)
            continue
        for piece in (_split_long(b.text, cfg.size) if _words(b.text) > cfg.size else [b.text]):
            n_cur = sum(_words(p) for p in cur)
            if cur and n_cur + _words(piece) > cfg.size:
                prev = cur
                flush_text()
                # overlap: carry trailing paragraphs of the previous chunk, up to `overlap` words
                carry, n = [], 0
                for p in reversed(prev):
                    if n + _words(p) > cfg.overlap:
                        break
                    carry.insert(0, p); n += _words(p)
                cur = carry
            cur.append(piece)
    flush_text()
    return chunks
