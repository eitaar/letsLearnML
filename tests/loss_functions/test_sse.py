import numpy as np

from letslearnml.loss_functions import sse


def test_sse():
    y = np.array([1])
    t = np.array([3])
    assert sse(y, t) == 2.0
