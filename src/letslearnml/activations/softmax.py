import numpy as np


def softmax(x: float) -> float:
    return np.exp(x) / np.sum(np.exp(x))
