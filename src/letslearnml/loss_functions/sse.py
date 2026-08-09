import numpy as np
from numpy.typing import NDArray


def sse(y: NDArray, t: NDArray) -> NDArray:
    return 0.5 * np.sum((y - t) ** 2)
