from rag_eval import generate as G

Q = {"type": "numeric", "question": "What was X's revenue?", "unit": "millions", "evidence": ["303,154"], "answer_value": 303154.0}


def test_parse_number_handles_commas_parens_and_dollar_signs():
    assert G.parse_number("Revenue was high.\nANSWER: 303154") == 303154
    assert G.parse_number("ANSWER: $303,154.5") == 303154.5
    assert G.parse_number("ANSWER: (1,200)") == -1200
    assert G.parse_number("no final line") is None


def test_numeric_grading_tolerance():
    assert G.numeric_correct("ANSWER: 303200", 303154, "millions")        # rounded from billions: within 0.5%
    assert not G.numeric_correct("ANSWER: 330000", 303154, "millions")    # 9% off
    assert G.numeric_correct("ANSWER: 4.50", 4.5, "dollars per share")
    assert not G.numeric_correct("ANSWER: 4.60", 4.5, "dollars per share")
    assert G.numeric_correct("ANSWER: 1200", -1200, "millions")           # loss written as a positive number
    assert not G.numeric_correct("NOT_FOUND", 5, "millions")


def test_abstention_detection():
    assert G.abstained("Not found")   # tolerant of casing and a space instead of the underscore
    assert G.abstained("NOT_FOUND") and not G.abstained("Revenue was 5 [1]")


def test_prompt_includes_numbered_excerpts_and_unit_for_numeric_only():
    sys_, user = G.build_prompt(Q, ["alpha", "beta"])
    assert "[1] alpha" in user and "[2] beta" in user and "ANSWER:" in user and "NOT_FOUND" in sys_
    _, user_text = G.build_prompt({"type": "text", "question": "Why?"}, ["alpha"])
    assert "ANSWER:" not in user_text
    sys_cb, user_cb = G.build_prompt(Q, None)
    assert "Excerpts" not in user_cb and "own knowledge" in sys_cb


def test_number_in_context_checks_accepted_strings():
    assert G.number_in_context(Q, ["net sales 303,154 total"]) and not G.number_in_context(Q, ["net sales 1,000"])
