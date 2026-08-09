import numpy as np

from letslearnml.loss_functions import sse

y = np.array([1])
t = np.array([3])

print(sse(y, t))
