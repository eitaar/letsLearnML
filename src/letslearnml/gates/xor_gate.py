from letslearnml.gates import AND, NAND, OR


def XOR(x1: int, x2: int, b: float | None = -0.2) -> int:
    l1a = NAND(x1, x2)
    l1b = OR(x1, x2)
    return AND(l1a, l1b)
