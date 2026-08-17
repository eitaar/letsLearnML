from collections import OrderedDict
from collections.abc import Sequence

import numpy as np

import letslearnml


class NLayerNet:
    def __init__(
        self,
        input_size: int,
        hidden_layer: int,
        hidden_size: Sequence[int],
        output_size: int,
        weight_init_std: float = 0.01,
        use_batch_norm: bool = True,
    ) -> None:
        self.hidden_layer_num = hidden_layer
        self.use_batch_norm = use_batch_norm
        self.params = {}
        for i in range(hidden_layer):
            layer_num = i + 1
            fan_in = input_size
            self.params[f"W{layer_num}"] = np.sqrt(2.0 / fan_in) * np.random.randn(input_size, hidden_size[i])
            self.params[f"b{layer_num}"] = np.zeros(hidden_size[i])
            if self.use_batch_norm:
                self.params[f"gamma{layer_num}"] = np.ones(hidden_size[i])
                self.params[f"beta{layer_num}"] = np.zeros(hidden_size[i])
            input_size = hidden_size[i]

        output_layer = hidden_layer + 1
        self.params[f"W{output_layer}"] = np.sqrt(2.0 / input_size) * np.random.randn(input_size, output_size)
        self.params[f"b{output_layer}"] = np.zeros(output_size)
        self.layers = OrderedDict()
        for i in range(hidden_layer):
            layer_num = i + 1
            self.layers[f"Affine{layer_num}"] = letslearnml.affines.Affine(
                self.params[f"W{layer_num}"], self.params[f"b{layer_num}"]
            )
            if self.use_batch_norm:
                self.layers[f"BatchNorm{layer_num}"] = letslearnml.batches.BatchNorm()
            self.layers[f"Relu{layer_num}"] = letslearnml.activations.Relu()

        self.layers[f"Affine{output_layer}"] = letslearnml.affines.Affine(
            self.params[f"W{output_layer}"], self.params[f"b{output_layer}"]
        )

        self.lastLayer = letslearnml.activations.SoftmaxWithLoss()

    def predict(self, x, is_training):
        for name, layer in self.layers.items():
            if name.startswith("BatchNorm"):
                layer_num = name.removeprefix("BatchNorm")
                x = layer.forward(x, self.params[f"gamma{layer_num}"], self.params[f"beta{layer_num}"], is_training)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x, True)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}

        for i in range(self.hidden_layer_num):
            layer_num = i + 1
            grads[f"W{layer_num}"] = letslearnml.gradients.numerical_gradient(loss_W, self.params[f"W{layer_num}"])
            grads[f"b{layer_num}"] = letslearnml.gradients.numerical_gradient(loss_W, self.params[f"b{layer_num}"])
            if self.use_batch_norm:
                grads[f"gamma{layer_num}"] = letslearnml.gradients.numerical_gradient(
                    loss_W, self.params[f"gamma{layer_num}"]
                )
                grads[f"beta{layer_num}"] = letslearnml.gradients.numerical_gradient(
                    loss_W, self.params[f"beta{layer_num}"]
                )

        layer_num = self.hidden_layer_num + 1
        grads[f"W{layer_num}"] = letslearnml.gradients.numerical_gradient(loss_W, self.params[f"W{layer_num}"])
        grads[f"b{layer_num}"] = letslearnml.gradients.numerical_gradient(loss_W, self.params[f"b{layer_num}"])

        return grads

    def gradient(self, x, t):
        # fwd
        self.loss(x, t)

        # bwd
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i in range(self.hidden_layer_num):
            layer_num = i + 1
            grads[f"W{layer_num}"], grads[f"b{layer_num}"] = (
                self.layers[f"Affine{layer_num}"].dW,
                self.layers[f"Affine{layer_num}"].db,
            )
            if self.use_batch_norm:
                grads[f"gamma{layer_num}"], grads[f"beta{layer_num}"] = (
                    self.layers[f"BatchNorm{layer_num}"].dgamma,
                    self.layers[f"BatchNorm{layer_num}"].dbeta,
                )

        layer_num = self.hidden_layer_num + 1
        grads[f"W{layer_num}"], grads[f"b{layer_num}"] = (
            self.layers[f"Affine{layer_num}"].dW,
            self.layers[f"Affine{layer_num}"].db,
        )
        return grads
