import numpy as np

from letslearnml.activations import Relu


def test_relu():
    ReLU = Relu()
    x = np.array([-2, 0, 3, 7])
    dout = np.array([10, 20, 30, 40])

    assert np.allclose(ReLU.forward(x), np.maximum(0, x))
    assert np.allclose(ReLU.backward(dout.copy()), np.where(x > 0, dout, 0))
