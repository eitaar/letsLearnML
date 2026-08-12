from numpy.typing import NDArray

from ..loss_functions.cel import cel
from .softmax import softmax


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t: NDArray

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cel(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
