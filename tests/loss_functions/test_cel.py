import numpy as np

from letslearnml.loss_functions import cel


def test_cel():
    y = np.array([0.8])
    t = np.array([1])

    assert np.isclose(cel(y, t), -np.log(0.8))
