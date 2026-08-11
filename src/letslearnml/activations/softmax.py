import numpy as np


def softmax(x):
    c = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
