import os
import sys

import numpy as np

sys.path.append(os.pardir)

from letslearnml import activations, gradients, loss_functions


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # init weight
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = weight_init_std * np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = activations.sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = activations.softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return loss_functions.cel(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads["W1"] = gradients.numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = gradients.numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = gradients.numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = gradients.numerical_gradient(loss_W, self.params["b2"])

        return grads
