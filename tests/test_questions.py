from rag_eval import questions as Q


def test_pick_value_flow_needs_a_one_year_duration_and_matching_end():
    rows = [{"end": "2025-12-31", "start": "2025-10-01", "val": 1},   # a quarter, not the year
            {"end": "2025-12-31", "start": "2025-01-01", "val": 4},
            {"end": "2024-12-31", "start": "2024-01-01", "val": 3}]
    assert Q.pick_value(rows, "2025-12-31", "flow") == 4
    assert Q.pick_value(rows, "2023-12-31", "flow") is None


def test_pick_value_instant_ignores_durations():
    rows = [{"end": "2025-12-31", "start": "2025-01-01", "val": 9}, {"end": "2025-12-31", "start": None, "val": 7}]
    assert Q.pick_value(rows, "2025-12-31", "instant") == 7


def test_accepted_strings_cover_table_and_prose_formats():
    assert Q.accepted_strings(303_154_000_000, "millions") == ["303,154", "303.2 billion"]
    assert Q.accepted_strings(-1_200_000_000, "millions") == ["1,200", "1.2 billion"]
    assert Q.accepted_strings(4.5, "dollars per share") == ["4.50"]


def test_prior_end_handles_leap_day():
    assert Q.prior_end("2024-02-29") == "2023-02-28" and Q.prior_end("2025-12-31", 5) == "2020-12-31"


def test_built_questions_name_the_company_and_have_unique_ids():
    qs = Q.build_numeric() + Q.build_unanswerable()
    assert len({q["id"] for q in qs}) == len(qs)
    assert all(any(n in q["question"] for n in ("Apple", "Microsoft", "JPMorgan", "Walmart", "Pfizer", "ExxonMobil", "Tesla", "Costco", "Nike", "Delta")) for q in qs)
