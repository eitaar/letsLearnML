import numpy as np

from letslearnml.gradients import numerical_gradient


def test_numerical_gradient_is_callable_from_package():
    gradient = numerical_gradient(lambda values: np.sum(values**2), np.array([1.0, 2.0]))

    assert np.allclose(gradient, np.array([2.0, 4.0]), atol=1e-4)


def test_numerical_gradient_supports_multidimensional_arrays():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    gradient = numerical_gradient(lambda current: np.sum(current**2), values)

    assert gradient.shape == values.shape
    assert np.allclose(gradient, 2 * values, atol=1e-4)
