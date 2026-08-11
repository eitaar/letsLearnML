import matplotlib.pyplot as plt
import numpy as np

from dataset.mnist import load_mnist
from TLnet import TwoLayerNet

print("Loading data...")
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

iters_num = 10
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

print("Building model...")
network = TwoLayerNet(784, 50, 10)

print("Training...")
count = 0
for i in range(iters_num):
    count += 1
    print(f"{count}/{iters_num}")
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    print("calculating grad...")
    # calc grad
    grad = network.numerical_gradient(x_batch, t_batch)

    print("upd param...")
    # upd param
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    print("calc loss...")
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
print("Done")

plt.plot(train_loss_list)
plt.show()
