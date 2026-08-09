from letslearnml.gates import AND


def test_and_gate():
    assert AND(0, 0) == 0
    assert AND(0, 1) == 0
    assert AND(1, 0) == 0
    assert AND(1, 1) == 1
