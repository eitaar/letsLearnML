import numpy as np

from letslearnml.activations import softmax


def test_softmax():
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    assert np.isclose(np.sum(result), 1.0)
    assert np.all(result >= 0)
    assert np.all(result <= 1)
    assert result[2] > result[1] > result[0]
