import numpy as np
from numpy.typing import NDArray


class Sigmoid:
    def __init__(self):
        self.out: float

    def forward(self, x: NDArray):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
