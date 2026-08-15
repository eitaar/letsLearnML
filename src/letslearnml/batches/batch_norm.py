import numpy as np


class BatchNorm:
    def __init__(self):
        self.x: float
        self.gamma: float
        self.beta: float
        self.dsigma: float
        self.dgamma: float
        self.x_cap: float
        self.eps = 1e-5
        self.mu: float
        self.sigma_squared: float
        self.sigma: float
        self.running_mean = None
        self.running_var = None
        self.momentum = 0.9

    def forward(self, x, gamma, beta, is_training: bool, eps=1e-5):
        self.eps = eps
        if self.running_mean is None or self.running_var is None:
            self.running_mean = np.zeros(x.shape[1])
            self.running_var = np.zeros(x.shape[1])

        if is_training:
            self.x = x
            self.gamma = gamma
            self.beta = beta

            self.mu = np.mean(x, axis=0)
            self.sigma_squared = np.mean((x - self.mu) ** 2, axis=0)
            self.sigma = np.sqrt(self.sigma_squared + self.eps)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sigma_squared

            self.x_cap = (x - self.mu) / self.sigma
            out = gamma * self.x_cap + beta
            return out
        else:
            mu = self.running_mean
            sigma_squared = self.running_var
            sigma = np.sqrt(sigma_squared + self.eps)
            x_cap = (x - mu) / sigma
            out = gamma * x_cap + beta
            return out

    def backward(self, dout):
        self.dgamma = np.sum(self.x_cap * dout, axis=0)
        self.dbeta = np.sum(dout, axis=0)
        dx_cap = dout * self.gamma
        dsigma_sq = np.sum(dx_cap * (self.x - self.mu), axis=0) * -0.5 * self.sigma**-3
        dmu = np.sum(-dx_cap / self.sigma, axis=0) + dsigma_sq * np.mean(-2 * (self.x - self.mu), axis=0)
        dx = dx_cap / self.sigma + dsigma_sq * (2 * (self.x - self.mu) / self.x.shape[0]) + dmu * 1 / self.x.shape[0]

        return dx
