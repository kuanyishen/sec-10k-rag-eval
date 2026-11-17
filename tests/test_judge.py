import random
from rag_eval import judge as J


def test_parse_verdict():
    assert J.parse_verdict('ok {"correct": true, "supported": false, "reason": "x"}')["supported"] is False
    assert J.parse_verdict('{"correct": "yes", "supported": true}') is None and J.parse_verdict("nope") is None


def test_corrupt_number_changes_value_keeps_format():
    rng = random.Random(0)
    out = J.corrupt_number("Revenue was $303,154 million [1].\nANSWER: 303154", rng)
    assert out is not None and "303,154" not in out.split("\n")[0] and "," in out.split("\n")[0]
    assert J.corrupt_number("No digits here", rng) is None


def test_agreement_and_kappa_by_hand():
    perfect = [(True, True)] * 5 + [(False, False)] * 5
    assert J.agreement(perfect)["kappa"] == 1.0
    a = J.agreement([(True, True), (True, False), (False, False), (False, False)])
    # accuracy 0.75; judge says True 50%, truth True 25% -> pe = .5*.25+.5*.75 = .5 -> kappa .5
    assert a["accuracy"] == 0.75 and abs(a["kappa"] - 0.5) < 1e-9 and a["false_accept_rate"] == 1 / 3
