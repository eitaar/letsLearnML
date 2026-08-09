from letslearnml.gates import NOR


def test_nor_gate():
    assert NOR(0, 0) == 1
    assert NOR(0, 1) == 0
    assert NOR(1, 0) == 0
    assert NOR(1, 1) == 0
