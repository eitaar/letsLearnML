import numpy as np
from numpy.typing import NDArray


def cel(y: NDArray, t: NDArray) -> NDArray:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + 1e-7))
