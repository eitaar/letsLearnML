import numpy as np


class BatchNorm:
    def __init__(self):
        self.x: float
        self.gamma: float
        self.beta: float
        self.x_cap: float
        self.eps = 1e-5
        self.mu: float
        self.sigma_squared: float
        self.sigma: float

    def forward(self, x, gamma, beta, eps=1e-5):
        self.x = x
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

        self.mu = np.mean(x, axis=0)
        self.sigma_squared = np.mean((x - self.mu) ** 2, axis=0)
        self.sigma = np.sqrt(self.sigma_squared + self.eps)

        self.x_cap = (x - self.mu) / self.sigma

        out = gamma * self.x_cap + beta
        return out

    def backward(self, dout):
        dgamma = np.sum(self.x_cap * dout, axis=0)
        dbeta = np.sum(dout, axis=0)
        dx_cap = dout * self.gamma
        dsigma_sq = np.sum(dx_cap * (self.x - self.mu), axis=0) * -0.5 * self.sigma**-3
        dmu = np.sum(-dx_cap / self.sigma, axis=0) + dsigma_sq * np.mean(-2 * (self.x - self.mu), axis=0)
        dx = dx_cap / self.sigma + dsigma_sq * (2 * (self.x - self.mu) / self.x.shape[0]) + dmu * 1 / self.x.shape[0]

        return dx
