import numpy as np

from letslearnml.activations import softmax


def test_softmax():
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    assert np.isclose(np.sum(result), 1.0)
    assert np.all(result >= 0)
    assert np.all(result <= 1)
    assert result[2] > result[1] > result[0]


def test_softmax_normalizes_each_sample_independently_for_a_batch():
    x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    result = softmax(x)

    # Each row is one sample, so its class probabilities must sum to 1.
    assert np.allclose(np.sum(result, axis=1), 1.0)
