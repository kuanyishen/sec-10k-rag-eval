"""Download each company's latest 10-K and its SEC structured (XBRL) facts.

Runs on GitHub Actions (the development sandbox cannot reach sec.gov):
    python -m rag_eval.edgar
SEC fair-access rules: identify yourself in the User-Agent and stay under 10 requests/second.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
# The SEC rejects anonymous clients (403). Set SEC_USER_AGENT to "Your Name contact@example.com".
def _clean(ua: str) -> str:
    """Collapse whitespace/newlines that a pasted secret may carry (they make the header invalid)."""
    return " ".join(ua.split())


UA = _clean(os.environ.get("SEC_USER_AGENT", "")) or "sec-10k-rag-eval research project (github.com/kuanyishen/sec-10k-rag-eval)"


def get(url: str, *, tries: int = 4) -> requests.Response:
    for i in range(tries):
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}, timeout=60)
        if r.status_code == 200:
            time.sleep(0.2)
            return r
        if r.status_code in (403, 429, 503) and i < tries - 1:
            time.sleep(2 ** (i + 1))
            continue
        if r.status_code == 403:
            raise SystemExit("SEC returned 403: set the SEC_USER_AGENT secret to 'Your Name your-contact@example.com' "
                             "(the SEC requires a real contact in the User-Agent).")
        r.raise_for_status()
    raise RuntimeError(url)


def latest_10k(submissions: dict) -> dict:
    """Most recent original 10-K (not 10-K/A) from the `recent` block of a submissions file."""
    rec = submissions["filings"]["recent"]
    rows = [dict(zip(rec.keys(), vals)) for vals in zip(*rec.values())]
    tenks = [r for r in rows if r["form"] == "10-K"]
    if not tenks:
        raise ValueError("no 10-K in recent filings")
    best = max(tenks, key=lambda r: (r["filingDate"], r["accessionNumber"]))
    return {"accession": best["accessionNumber"], "filing_date": best["filingDate"],
            "report_date": best["reportDate"], "primary_document": best["primaryDocument"]}


def extract_facts(companyfacts: dict, accession: str, concepts: dict[str, list[str]]) -> dict:
    """Facts reported inside this exact filing: {label: {concept, unit, rows:[{end,start,val,fy,fp}]}}.
    Includes prior-year comparatives printed in the same filing; consumers pick by period."""
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    out = {}
    for label, names in concepts.items():
        for name in names:
            units = gaap.get(name, {}).get("units", {})
            for unit, items in units.items():
                rows = [{"end": i["end"], "start": i.get("start"), "val": i["val"], "fy": i.get("fy"), "fp": i.get("fp")}
                        for i in items if i.get("accn") == accession and i.get("form") == "10-K"]
                if rows:
                    out[label] = {"concept": name, "unit": unit, "rows": rows}
                    break
            if label in out:
                break
    return out


def collect(cfg_path: Path = ROOT / "config" / "companies.yaml") -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in cfg["companies"]:
        cik10 = f"{c['cik']:010d}"
        sub = get(f"https://data.sec.gov/submissions/CIK{cik10}.json").json()
        if c["expect"].lower() not in sub["name"].lower():
            raise SystemExit(f"CIK {c['cik']} is registered as {sub['name']!r}, expected {c['expect']!r}")
        f = latest_10k(sub)
        acc_nd = f["accession"].replace("-", "")
        html = get(f"https://www.sec.gov/Archives/edgar/data/{c['cik']}/{acc_nd}/{f['primary_document']}").content
        (RAW / f"{c['ticker']}_10k.htm").write_bytes(html)
        facts = extract_facts(get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json").json(),
                              f["accession"], cfg["concepts"])
        (RAW / f"{c['ticker']}_facts.json").write_text(json.dumps(facts, indent=1))
        manifest.append({**c, "name": sub["name"], **f, "html_bytes": len(html), "facts_found": sorted(facts)})
        print(c["ticker"], sub["name"], f["report_date"], f"{len(html)/1e6:.1f}MB", "facts:", len(facts))
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    collect()
