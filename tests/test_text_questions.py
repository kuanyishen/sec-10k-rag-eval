from rag_eval.gen_text_questions import evidence_is_verbatim, parse_reply

P = "Our   business is seasonal.  Sales in the fourth quarter are\nhistorically the highest of the year."


def test_verbatim_check_ignores_whitespace_but_not_wording():
    assert evidence_is_verbatim("Sales in the fourth quarter are historically the highest of the year.", P)
    assert not evidence_is_verbatim("Sales in the fourth quarter tend to be the highest of the year.", P)
    assert not evidence_is_verbatim("Our business", P)  # too short to be meaningful evidence


def test_parse_reply_extracts_json_from_chatter_and_rejects_incomplete():
    ok = 'Here you go: {"question": "q?", "answer": "a", "evidence": "e e e e e", "qtype": "fact"} thanks'
    assert parse_reply(ok)["qtype"] == "fact"
    assert parse_reply('{"question": "q?"}') is None and parse_reply("no json") is None
