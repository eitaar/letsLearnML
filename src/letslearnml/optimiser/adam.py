import numpy as np


# Adam optimiser
class Adam:
    def __init__(self, beta1, beta2, lr=0.01, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        if (self.m is None) or (self.v is None):
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        for key in params:
            self.m[key] = (self.beta1 * self.m[key]) + (1 - self.beta1) * grads[key]
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            self.v[key] = (self.beta2 * self.v[key]) + (1 - self.beta2) * np.square(grads[key])
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            params[key] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))
