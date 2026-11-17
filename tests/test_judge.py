import random
from rag_eval import judge as J


def test_parse_verdict():
    assert J.parse_verdict('ok {"correct": true, "supported": false, "reason": "x"}')["supported"] is False
    assert J.parse_verdict('{"correct": "yes", "supported": true}') is None and J.parse_verdict("nope") is None


def test_corrupt_answer_changes_every_form_of_the_value_but_not_dates():
    a = "Apple's net sales for the fiscal year ended September 26, 2026 were $478,834 million ($478.8 billion) [4].\n\nANSWER: 478834"
    c = J.corrupt_answer(a, 478834.0, "millions", 1.2)
    assert c is not None and "478,834" not in c and "478834" not in c and "478.8 billion" not in c
    assert "September 26, 2026" in c and "ANSWER: 574601" in c
    assert J.corrupt_answer("ANSWER: 8.98 per share", 8.98, "dollars per share", 0.8) == "ANSWER: 7.18 per share"
    assert J.corrupt_answer("NOT_FOUND", 5.0, "millions", 1.2) is None   # nothing to corrupt -> no probe


def test_agreement_and_kappa_by_hand():
    perfect = [(True, True)] * 5 + [(False, False)] * 5
    assert J.agreement(perfect)["kappa"] == 1.0
    a = J.agreement([(True, True), (True, False), (False, False), (False, False)])
    # accuracy 0.75; judge says True 50%, truth True 25% -> pe = .5*.25+.5*.75 = .5 -> kappa .5
    assert a["accuracy"] == 0.75 and abs(a["kappa"] - 0.5) < 1e-9 and a["false_accept_rate"] == 1 / 3
