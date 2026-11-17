"""Questions whose answers are checkable without an LLM.

numeric      - answers come from the SEC's structured XBRL facts for this exact filing
unanswerable - asks for periods the filing does not report; the right behaviour is to abstain
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

TEMPLATES = {  # label -> (question template, kind of period, unit)
    "revenue": ("What were {name}'s total revenues (net sales) for the fiscal year ended {end}?", "flow", "millions"),
    "net_income": ("What was {name}'s net income for the fiscal year ended {end}?", "flow", "millions"),
    "operating_income": ("What was {name}'s operating income for the fiscal year ended {end}?", "flow", "millions"),
    "operating_cash_flow": ("How much net cash did {name} provide from operating activities in the fiscal year ended {end}?", "flow", "millions"),
    "rd_expense": ("How much did {name} spend on research and development in the fiscal year ended {end}?", "flow", "millions"),
    "diluted_eps": ("What was {name}'s diluted earnings per share for the fiscal year ended {end}?", "flow", "dollars per share"),
    "total_assets": ("What were {name}'s total assets as of {end}?", "instant", "millions"),
    "total_liabilities": ("What were {name}'s total liabilities as of {end}?", "instant", "millions"),
    "stockholders_equity": ("What was {name}'s total stockholders' equity as of {end}?", "instant", "millions"),
    "cash": ("How much cash and cash equivalents did {name} have as of {end}?", "instant", "millions"),
}


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def pick_value(rows: list[dict], end: str, kind: str):
    """The fact for the period ending `end`: a ~1-year duration for flows, a point in time for balances."""
    for r in rows:
        if r["end"] != end:
            continue
        if kind == "flow" and r.get("start") and 340 <= _days(r["start"], r["end"]) <= 380:
            return r["val"]
        if kind == "instant" and not r.get("start"):
            return r["val"]
    return None


def accepted_strings(val: float, unit: str) -> list[str]:
    """Ways the number is printed in a filing: '303,154' (millions, in tables), '303.2 billion' (prose)."""
    if unit == "dollars per share":
        return [f"{abs(val):.2f}"]
    m = abs(val) / 1e6
    out = [f"{m:,.0f}"]
    if m >= 1000:
        out.append(f"{m / 1000:,.1f} billion")
    return out


def prior_end(end: str, years: int = 1) -> str:
    d = date.fromisoformat(end)
    try:
        return d.replace(year=d.year - years).isoformat()
    except ValueError:  # Feb 29
        return d.replace(year=d.year - years, day=28).isoformat()


def _fmt(end: str) -> str:
    return date.fromisoformat(end).strftime("%B %-d, %Y")


def build_numeric(root: Path = ROOT) -> list[dict]:
    cfg = yaml.safe_load((root / "config" / "companies.yaml").read_text())
    man = {m["ticker"]: m for m in json.loads((root / "data" / "raw" / "manifest.json").read_text())}
    qs = []
    for c in cfg["companies"]:
        t, report_end = c["ticker"], man[c["ticker"]]["report_date"]
        facts = json.loads((root / "data" / "raw" / f"{t}_facts.json").read_text())
        for label, (tmpl, kind, unit) in TEMPLATES.items():
            if label not in facts:
                continue
            rows = facts[label]["rows"]
            for which, end in (("current", report_end), ("prior", prior_end(report_end))):
                v = pick_value(rows, end, kind)
                if v is None:
                    continue
                qs.append({"id": f"num-{t}-{label}-{which}", "type": "numeric", "ticker": t, "metric": label, "period": which,
                           "question": tmpl.format(name=c["short"], end=_fmt(end)), "unit": unit,
                           "answer_value": v / 1e6 if unit == "millions" else v,
                           "evidence": accepted_strings(v, unit)})
    return qs


def build_unanswerable(root: Path = ROOT) -> list[dict]:
    """Periods the filing cannot report: five years earlier, and the year after the filing's own year."""
    cfg = yaml.safe_load((root / "config" / "companies.yaml").read_text())
    man = {m["ticker"]: m for m in json.loads((root / "data" / "raw" / "manifest.json").read_text())}
    qs = []
    for c in cfg["companies"]:
        t, end = c["ticker"], man[c["ticker"]]["report_date"]
        d = date.fromisoformat(end)
        nxt = d.replace(year=d.year + 1).isoformat()
        old = prior_end(end, 5)
        for tag, metric, tmpl, e in (
            ("future-revenue", "revenue", "What were {name}'s total revenues for the fiscal year ended {end}?", nxt),
            ("old-net-income", "net_income", "What was {name}'s net income for the fiscal year ended {end}?", old),
            ("future-net-income", "net_income", "What was {name}'s net income for the fiscal year ended {end}?", nxt),
        ):
            qs.append({"id": f"una-{t}-{tag}", "type": "unanswerable", "ticker": t, "metric": metric,
                       "question": tmpl.format(name=c["short"], end=_fmt(e)), "reason": "period not reported in this filing"})
    return qs


if __name__ == "__main__":
    out = ROOT / "data" / "questions"
    out.mkdir(parents=True, exist_ok=True)
    for name, qs in (("numeric", build_numeric()), ("unanswerable", build_unanswerable())):
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(q) for q in qs) + "\n")
        print(name, len(qs))
