import numpy as np


class BatchNorm:
    def __init__(self):
        self.x = None
        self.gamma = None
        self.beta = None
        self.eps = 1e-5

    def forward(self, x, gamma, beta, eps=1e-5):
        self.x = x
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

        batch_mean = np.mean(x, axis=0)
        sigma_squared = np.mean((x - batch_mean) ** 2, axis=0)
        sigma = np.sqrt(sigma_squared + self.eps)

        x_cap = (x - batch_mean) / sigma

        out = gamma * x_cap + beta
        return out

    def backward(self, dout):
        pass
