import numpy as np
from numpy.typing import NDArray


def numerical_gradient(f, x: NDArray):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        tmp_val = x[idx]
        # calculate f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # calculate f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad
