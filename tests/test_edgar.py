import pytest
from rag_eval import edgar

SUB = {"filings": {"recent": {
    "form": ["8-K", "10-K", "10-K/A", "10-K"],
    "accessionNumber": ["a-1", "a-2", "a-3", "a-0"],
    "filingDate": ["2025-12-01", "2025-10-31", "2025-11-15", "2024-10-30"],
    "reportDate": ["", "2025-09-27", "2025-09-27", "2024-09-28"],
    "primaryDocument": ["x.htm", "k25.htm", "ka.htm", "k24.htm"]}}}


def test_latest_10k_picks_newest_original_not_amendment():
    f = edgar.latest_10k(SUB)
    assert f["accession"] == "a-2" and f["primary_document"] == "k25.htm" and f["report_date"] == "2025-09-27"


def test_latest_10k_requires_a_10k():
    with pytest.raises(ValueError):
        edgar.latest_10k({"filings": {"recent": {"form": ["8-K"], "accessionNumber": ["x"], "filingDate": ["d"],
                                                  "reportDate": [""], "primaryDocument": ["p"]}}})


def test_extract_facts_keeps_only_this_filing_and_first_matching_concept():
    cf = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"end": "2025-09-27", "start": "2024-09-29", "val": 100, "accn": "a-2", "form": "10-K", "fy": 2025, "fp": "FY"},
            {"end": "2024-09-28", "start": "2023-10-01", "val": 90, "accn": "a-2", "form": "10-K", "fy": 2025, "fp": "FY"},
            {"end": "2025-06-28", "val": 70, "accn": "q-3", "form": "10-Q", "fy": 2025, "fp": "Q3"}]}},
        "SalesRevenueNet": {"units": {"USD": [{"end": "2025-09-27", "val": 1, "accn": "a-2", "form": "10-K"}]}}}}}
    out = edgar.extract_facts(cf, "a-2", {"revenue": ["Revenues", "SalesRevenueNet"], "net_income": ["NetIncomeLoss"]})
    assert out["revenue"]["concept"] == "Revenues" and len(out["revenue"]["rows"]) == 2
    assert "net_income" not in out
    assert edgar.extract_facts(cf, "other", {"revenue": ["Revenues"]}) == {}


def test_user_agent_cleaning_removes_newlines_and_edge_whitespace():
    assert edgar._clean("  Sky Shen \n a@b.com \r\n") == "Sky Shen a@b.com"
