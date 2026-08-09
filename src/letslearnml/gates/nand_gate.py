import numpy as np


def NAND(x1: int, x2: int, b: float | None = -0.7) -> int:
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    tmp = np.sum(w * x) + b

    if tmp <= 0:
        return 1
    else:
        return 0
