import numpy as np


def cel(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
