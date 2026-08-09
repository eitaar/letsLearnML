import random

import numpy as np

from letslearnml.activations import ReLU


def test_relu():
    randint = random.randint(1, 10)
    assert ReLU(-1) == 0
    assert ReLU(0) == 0
    assert ReLU(1) == 1
    assert ReLU(randint) == randint
    assert np.allclose(ReLU(np.array([-1, 0, 1, 2])), np.array([0, 0, 1, 2]))
