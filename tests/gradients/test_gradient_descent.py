import numpy as np

from letslearnml.gradients import gradient_descent


def test_gradient_descent():
    init_x = np.array([-3.0, 4.0])
    assert np.allclose(
        gradient_descent(lambda values: values[0] ** 2 + values[1] ** 2, init_x=init_x, lr=0.1, step_num=100),
        np.array([-6.11e-10, 8.15e-10]),
        1e-10,
    )
