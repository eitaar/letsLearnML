"""MNIST training with an eight-process numerical-gradient calculation."""

import os

# Numerical gradients are CPU-bound. Keep BLAS from creating extra threads in
# every worker process, otherwise 8 processes can oversubscribe the CPU.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from dataset.mnist import load_mnist
from TLnet import TwoLayerNet

WORKERS = min(8, os.cpu_count() or 1)
_PARAMETER_KEYS = ("W1", "b1", "W2", "b2")


def _gradient_chunk(task):
    """Calculate one chunk of a numerical gradient in a worker process."""
    network, x, t, coordinates = task
    partial = {key: np.zeros_like(network.params[key]) for key in _PARAMETER_KEYS}
    h = 1e-4

    for key, flat_index in coordinates:
        parameter = network.params[key]
        index = np.unravel_index(flat_index, parameter.shape)
        original = parameter[index]

        try:
            parameter[index] = original + h
            fxh1 = network.loss(x, t)

            parameter[index] = original - h
            fxh2 = network.loss(x, t)

            partial[key][index] = (fxh1 - fxh2) / (2 * h)
        finally:
            parameter[index] = original

    return partial


def _parameter_coordinates(network):
    """Return every parameter coordinate in a balanced round-robin order."""
    return [(key, flat_index) for key in _PARAMETER_KEYS for flat_index in range(network.params[key].size)]


def parallel_numerical_gradient(network, x, t, workers=WORKERS):
    """Calculate numerical gradients using multiple CPU processes."""
    if workers < 1:
        raise ValueError("workers must be at least 1")

    workers = min(workers, os.cpu_count() or 1)
    coordinates = _parameter_coordinates(network)
    chunks = [coordinates[index::workers] for index in range(workers)]
    tasks = [(network, x, t, chunk) for chunk in chunks if chunk]

    gradients = {key: np.zeros_like(network.params[key]) for key in _PARAMETER_KEYS}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for partial in executor.map(_gradient_chunk, tasks):
            for key in _PARAMETER_KEYS:
                gradients[key] += partial[key]

    return gradients


def main():
    print(f"Using {WORKERS} worker processes")
    print("Loading data...")
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    iters_num = 2000
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
        print(f"{i + 1}/{iters_num}")
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        print("calculating gradient...")
        gradients = parallel_numerical_gradient(network, x_batch, t_batch)

        print("updating parameters...")
        for key in _PARAMETER_KEYS:
            network.params[key] -= learning_rate * gradients[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # Record at the end of each epoch, and also record a short debug run's
        # final iteration when iters_num is smaller than one full epoch.
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"train acc, test acc | {train_acc}, {test_acc}")

    print("Done")
    print(f"train acc, test acc: {train_acc}, {test_acc}")
    plt.plot(train_acc_list, label="train acc")
    plt.plot(test_acc_list, label="test acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
