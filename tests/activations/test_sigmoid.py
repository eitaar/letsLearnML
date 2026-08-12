import numpy as np

from letslearnml.activations import Sigmoid


def test_sigmoid_forward():
    sigmoid = Sigmoid()
    x = np.array([1.0, 0.0, -1.0])

    expected = np.array([0.7310586, 0.5, 0.2689414])

    assert np.allclose(sigmoid.forward(x), expected)


def test_sigmoid_backward():
    sigmoid = Sigmoid()
    x = np.array([1.0, 0.0, -1.0])
    dout = np.array([1.0, 2.0, 3.0])

    out = sigmoid.forward(x)
    expected = dout * out * (1.0 - out)

    assert np.allclose(sigmoid.backward(dout), expected)
