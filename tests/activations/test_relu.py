import numpy as np

from letslearnml.activations import ReLU as ReLU_Class


def test_relu():
    relu = ReLU_Class()
    x = np.array([-2, 0, 3, 7])
    dout = np.array([10, 20, 30, 40])

    assert np.allclose(relu.forward(x), np.maximum(0, x))
    assert np.allclose(relu.backward(dout.copy()), np.where(x > 0, dout, 0))
