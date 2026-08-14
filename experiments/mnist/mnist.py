import matplotlib.pyplot as plt
import numpy as np

from dataset.mnist import load_mnist
from experiments.models.TLnet import TwoLayerNet

print("Loading data...")
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

iters_num = 10
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

print("Building model...")
network = TwoLayerNet(784, 50, 10)

print("Training...")
for i in range(iters_num):
    print(f"{i}/{iters_num}")
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

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"train acc, test acc | {train_acc!s}, {test_acc!s}")
print("Done")

# draw accuracy on y and epoch on x graph using matplotlib
plt.plot(train_acc_list, label="train acc")
plt.plot(test_acc_list, label="test acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc="lower right")
plt.show()
