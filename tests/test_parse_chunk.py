from rag_eval.chunk import ChunkCfg, chunk_blocks
from rag_eval.parse import Block, blocks_from_html, render_table
from lxml import html as lh

HTML = b"""<html><body>
<ix:header><xbrli:context>HIDDEN 12345</xbrli:context></ix:header>
<div>Table of Contents</div>
<div>Item 1. Business</div><div>12</div>
<div>Item 7. Management Discussion</div><div>40</div>
<div><span>Item 1. Business</span></div>
<div>We sell <span>widgets</span> worldwide.</div><div>See Item 7 of this report.</div>
<div>Item 7. Management's Discussion and Analysis</div>
<div>Net sales rose.</div>
<table><tr><td>Net sales</td><td>$</td><td>1,234</td><td>$</td><td>(5</td><td>)</td></tr>
<tr><td>Margin</td><td></td><td>12.5</td><td>%</td></tr></table>
</body></html>"""


def test_blocks_separate_paragraphs_drop_hidden_xbrl_and_label_sections():
    bl = blocks_from_html(HTML)
    texts = [b.text for b in bl]
    assert not any("HIDDEN" in t for t in texts)
    assert "We sell widgets worldwide." in texts
    by = {b.text: b.section for b in bl}
    assert by["We sell widgets worldwide."] == "Item 1 Business"
    assert by["See Item 7 of this report."] == "Item 1 Business"  # an inline reference must not start a section
    assert by["Net sales rose."] == "Item 7 MD&A"


def test_table_rendering_glues_currency_fragments():
    t = lh.fromstring("<table><tr><td>Net sales</td><td>$</td><td>1,234</td></tr><tr><td>Margin</td><td>12.5</td><td>%</td></tr></table>")
    assert render_table(t) == "Net sales | $1,234\nMargin | 12.5%"


def test_page_number_lines_are_dropped():
    assert "12" not in [b.text for b in blocks_from_html(HTML)]


def _text_blocks(n_par, words=30, section="Item 1 Business"):
    return [Block("text", " ".join(f"w{p}_{i}" for i in range(words)), section) for p in range(n_par)]


def test_chunks_respect_size_and_carry_overlap():
    cfg = ChunkCfg(size=100, overlap=40)
    ch = chunk_blocks(_text_blocks(10), "T", "Test Co | FY2025", cfg)
    assert all(c.n_words <= 100 for c in ch) and len(ch) > 1
    assert ch[0].body.split("\n")[-1] in ch[1].body  # last paragraph repeated in the next chunk


def test_zero_overlap_means_no_repeated_paragraph():
    ch = chunk_blocks(_text_blocks(10), "T", "x", ChunkCfg(size=100, overlap=0))
    assert ch[0].body.split("\n")[-1] not in ch[1].body


def test_header_toggle_changes_indexed_text_only():
    b = _text_blocks(2)
    with_h = chunk_blocks(b, "T", "Costco | FY2026", ChunkCfg(header=True))
    no_h = chunk_blocks(b, "T", "Costco | FY2026", ChunkCfg(header=False))
    assert with_h[0].text.startswith("[Costco | FY2026 | Item 1 Business]") and no_h[0].text == no_h[0].body
    assert with_h[0].body == no_h[0].body


def test_sections_never_mix_in_a_chunk():
    b = _text_blocks(2, section="Item 1 Business") + _text_blocks(2, section="Item 7 MD&A")
    ch = chunk_blocks(b, "T", "x", ChunkCfg(size=500))
    assert {c.section for c in ch} == {"Item 1 Business", "Item 7 MD&A"} and len(ch) == 2


def test_long_table_split_repeats_header_rows():
    rows = ["Metric | 2025 | 2024", "-- | -- | --"] + [f"row{i} | {i} | {i}" for i in range(200)]
    ch = chunk_blocks([Block("table", "\n".join(rows), "Item 8 Financial Statements")], "T", "x", ChunkCfg(size=100, header=False))
    assert len(ch) > 1 and all(c.body.startswith("Metric | 2025 | 2024") for c in ch) and all(c.kind == "table" for c in ch)


def test_one_row_table_heading_is_a_section_heading_but_a_toc_table_is_not():
    h = b"""<html><body>
    <table><tr><td>Item 1</td><td>Business</td><td>6</td></tr><tr><td>Item 1A</td><td>Risk Factors</td><td>13</td></tr></table>
    <table><tr><td>ITEM 1A.</td><td>RISK FACTORS</td></tr></table>
    <div>Our risks are many.</div></body></html>"""
    bl = blocks_from_html(h)
    assert [b.section for b in bl if b.text == "Our risks are many."] == ["Item 1A Risk Factors"]
    assert sum(b.kind == "table" for b in bl) == 1  # only the multi-row TOC stays a table


def test_long_item_heading_is_recognised_and_big_appended_item15_is_renamed():
    long_h = "Item 7\u2014Management's Discussion and Analysis of Financial Condition and Results of Operations (amounts in millions, except per share data)"
    big = " ".join(["word"] * 6000)
    h = f"<html><body><div>{long_h}</div><div>MD&A text.</div><div>Item 15. Exhibits</div><div>{big}</div></body></html>".encode()
    by = {b.text[:12]: b.section for b in blocks_from_html(h)}
    assert by["MD&A text."] == "Item 7 MD&A" and by["word word wo"] == "Financial section (appended)"
