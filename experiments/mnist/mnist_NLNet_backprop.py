from collections.abc import Sequence

import numpy as np

from dataset.mnist import load_mnist
from experiments.models.NLnet import NLayerNet
from letslearnml.optimiser import Adam


def mnist(hidden_layer: int, hidden_size: Sequence[int]):
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = NLayerNet(784, hidden_layer, hidden_size, 10)

    iters_num = 1801
    train_size = x_train.shape[0]
    batch_size = 100
    learing_rate = 0.01

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    optimiser = Adam(beta1=0.9, beta2=0.999)
    for i in range(iters_num):
        print(f"{i}/{iters_num}, {(i / iters_num) * 100}%")
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        optimiser.update(network.params, grad)

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)

    # # draw accuracy on y and epoch on x graph using matplotlib
    # plt.plot(train_acc_list, label="train acc")
    # plt.plot(test_acc_list, label="test acc")
    # plt.xlabel("epoch")
    # plt.ylabel("accuracy")
    # plt.ylim(0, 1.0)
    # plt.legend(loc="lower right")
    # plt.show()

    # return the latest accuracy
    return train_acc_list[-1], test_acc_list[-1]
