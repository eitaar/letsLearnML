import numpy as np


def OR(x1: int, x2: int, b: float | None = -0.2) -> int:
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    tmp = np.sum(w * x) + b

    if tmp <= 0:
        return 0
    else:
        return 1
