from letslearnml.gates import OR


def test_or_gate():
    assert OR(0, 0) == 0
    assert OR(0, 1) == 1
    assert OR(1, 0) == 1
    assert OR(1, 1) == 1
