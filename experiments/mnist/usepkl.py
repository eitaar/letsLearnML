import pickle
from pathlib import Path

import numpy as np

from dataset.mnist import load_mnist
from experiments.models.NLnet import NLayerNet

# pklを読み込む
pkl_path = Path(__file__).resolve().parents[2] / "mnist_NLnet_backprop.pkl"

with pkl_path.open("rb") as f:
    data = pickle.load(f)


network = NLayerNet(
    input_size=784,
    hidden_layer=data["hidden_layer"],
    hidden_size=data["hidden_size"],
    output_size=10,
    use_batch_norm=data["use_batch_norm"],
)


for key, value in data["params"].items():
    network.params[key][...] = value

if network.use_batch_norm:
    for layer_name, stats in data["batch_norm_stats"].items():
        layer = network.layers[layer_name]
        layer.running_mean = stats["running_mean"]
        layer.running_var = stats["running_var"]


(_, _), (x_test, t_test) = load_mnist(
    normalize=True,
    one_hot_label=False,
)
randint = np.random.randint(x_test.shape[0])
x = x_test[randint : randint + 1]
print(x.shape)

scores = network.predict(x, is_training=False)
answer = np.argmax(scores, axis=1)

print("予測:", answer[0])
print("正解:", t_test[randint])
