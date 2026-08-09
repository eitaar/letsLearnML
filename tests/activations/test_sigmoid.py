import numpy as np

from letslearnml.activations import sigmoid


def test_sigmoid():
    assert round(sigmoid(1.0), 3) == 0.731
    assert round(sigmoid(0), 3) == 0.5
    assert round(sigmoid(-1.0), 3) == 0.269
    assert np.allclose(sigmoid(np.array([1.0, 0, -1.0])), np.array([0.731, 0.5, 0.269]), atol=0.001)
