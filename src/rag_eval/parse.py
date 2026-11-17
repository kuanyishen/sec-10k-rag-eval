"""10-K HTML -> ordered blocks (paragraphs and tables) tagged with their Item section."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from lxml import html

sys.setrecursionlimit(10000)

BLOCK_TAGS = {"div", "p", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
              "body", "tr", "center", "blockquote", "pre"}
SKIP_TAGS = {"script", "style", "ix:header", "head"}
ITEM_RE = re.compile(r"^item\s+(\d{1,2}[abc]?)\s*[.:—–\-]?\s*(.*)$", re.I)
ITEM_NAMES = {"1": "Business", "1a": "Risk Factors", "1b": "Unresolved Staff Comments", "1c": "Cybersecurity",
              "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety", "5": "Market for Common Equity",
              "6": "Reserved", "7": "MD&A", "7a": "Market Risk", "8": "Financial Statements",
              "9": "Disagreements with Accountants", "9a": "Controls and Procedures", "9b": "Other Information",
              "9c": "Foreign Jurisdictions", "10": "Directors and Governance", "11": "Executive Compensation",
              "12": "Security Ownership", "13": "Relationships and Transactions", "14": "Accountant Fees",
              "15": "Exhibits and Financial Statement Schedules", "16": "Form 10-K Summary"}


@dataclass
class Block:
    kind: str      # "text" or "table"
    text: str
    section: str   # e.g. "Item 7 MD&A", or "Front matter"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ").replace("​", " ")).strip()


def _cell_text(td) -> str:
    return _norm(td.text_content())


def render_table(tbl) -> str:
    """Rows become 'a | b | c' lines. Currency/percent/paren fragments are glued to their number."""
    lines = []
    for tr in tbl.iter("tr"):
        cells = [_cell_text(c) for c in tr if isinstance(c.tag, str) and c.tag in ("td", "th")]
        cells = [c for c in cells if c]
        merged: list[str] = []
        for c in cells:
            if merged and (c in {")", "%", ")%"} or merged[-1] in {"$", "(", "$("}):
                merged[-1] = merged[-1] + c if merged[-1] in {"$", "(", "$("} else merged[-1] + c
            else:
                merged.append(c)
        if merged:
            lines.append(" | ".join(merged))
    return "\n".join(lines)


def _walk(el, out: list, buf: list):
    def flush():
        t = _norm("".join(buf))
        buf.clear()
        if t:
            out.append(("text", t))

    tag = el.tag if isinstance(el.tag, str) else ""
    if tag in SKIP_TAGS or "display:none" in (el.get("style") or "").replace(" ", ""):
        return
    if tag == "table":
        flush()
        t = render_table(el)
        if t:
            out.append(("table", t))
        return
    is_block = tag in BLOCK_TAGS
    if is_block:
        flush()
    if tag == "br":
        flush()
    if el.text:
        buf.append(el.text)
    for ch in el:
        _walk(ch, out, buf)
        if ch.tail:
            buf.append(ch.tail)
    if is_block:
        flush()


_JUNK = re.compile(r"^(\d{1,3}|table of contents|page \d+( of \d+)?|\|)$", re.I)


def blocks_from_html(raw: bytes) -> list[Block]:
    doc = html.fromstring(raw)
    out: list[tuple[str, str]] = []
    _walk(doc, out, [])
    blocks: list[Block] = []
    section = "Front matter"
    for kind, text in out:
        if kind == "table" and "\n" not in text:
            flat = text.replace(" | ", " ")
            if ITEM_RE.match(flat) and len(flat) <= 220:   # some filers set headings inside a one-row table
                kind, text = "text", flat
        if kind == "text":
            if _JUNK.match(text):
                continue
            m = ITEM_RE.match(text)
            if m and len(text) <= 220 and not re.match(r"(of|in|to|and|for)\b", m.group(2), re.I):
                key = m.group(1).lower()
                section = f"Item {key.upper()} {ITEM_NAMES.get(key, m.group(2)[:40])}".strip()
                # a bare heading line is kept as text so the section name is searchable
        blocks.append(Block(kind, text, section))
    # Some filers (JPMorgan, Exxon) append the whole financial report after "Item 15/16".
    words: dict[str, int] = {}
    for b in blocks:
        words[b.section] = words.get(b.section, 0) + len(b.text.split())
    for b in blocks:
        if b.section.startswith(("Item 15", "Item 16")) and words[b.section] > 5000:
            b.section = "Financial section (appended)"
    return blocks
