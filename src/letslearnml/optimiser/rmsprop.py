import numpy as np


class RMSProp:
    def __init__(self, beta, lr=0.01, eps=1e-8):
        self.beta = beta
        self.lr = lr
        self.eps = eps
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            # v = βv + (1-β)dw^2
            self.v[key] = self.beta * self.v[key] + (1 - self.beta) * np.square(grads[key])
            # W += W - η * dw / (sqrt(v) + ε)
            params[key] -= self.lr * grads[key] / (np.sqrt(self.v[key]) + self.eps)
