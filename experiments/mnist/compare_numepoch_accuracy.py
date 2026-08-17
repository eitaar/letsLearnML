from collections.abc import Mapping, Sequence

import numpy as np

from dataset.mnist import load_mnist
from experiments.models.NLnet import NLayerNet
from letslearnml.optimiser import Adam

EPOCHS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    15,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
)
BATCH_SIZE = 100
LEARNING_RATE = 0.001
HIDDEN_LAYER = 5
HIDDEN_SIZE = (512, 256, 128, 64, 32)


def iterations_for_epochs(
    epochs: Sequence[int],
    train_size: int,
    batch_size: int = BATCH_SIZE,
) -> list[int]:
    iterations_per_epoch = max(train_size // batch_size, 1)
    return [epoch * iterations_per_epoch for epoch in epochs]


def create_network(use_batch_norm: bool = True) -> NLayerNet:
    return NLayerNet(784, HIDDEN_LAYER, HIDDEN_SIZE, 10, use_batch_norm=use_batch_norm)


def compare_epoch_accuracy(
    epochs: Sequence[int] = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = 0,
) -> dict[bool, tuple[list[float], list[float]]]:
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    iterations_per_epoch = max(x_train.shape[0] // batch_size, 1)
    checkpoint_epochs = set(epochs)
    max_epoch = max(epochs)
    results = {}

    for use_batch_norm in (True, False):
        train_accuracies = []
        test_accuracies = []
        condition = "with BatchNorm" if use_batch_norm else "without BatchNorm"

        np.random.seed(seed)
        random_generator = np.random.default_rng(seed)
        network = create_network(use_batch_norm=use_batch_norm)
        optimiser = Adam(beta1=0.9, beta2=0.999, lr=LEARNING_RATE)

        print(f"Training {condition} through {max_epoch} epoch(s)")
        for epoch in range(1, max_epoch + 1):
            for iteration in range(iterations_per_epoch):
                batch_mask = random_generator.choice(x_train.shape[0], batch_size)
                x_batch = x_train[batch_mask]
                t_batch = t_train[batch_mask]

                gradients = network.gradient(x_batch, t_batch)
                optimiser.update(network.params, gradients)

            if epoch in checkpoint_epochs:
                train_accuracy = network.accuracy(x_train, t_train)
                test_accuracy = network.accuracy(x_test, t_test)
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)
                print(f"  epoch {epoch}: train accuracy: {train_accuracy:.4f}, test accuracy: {test_accuracy:.4f}")

        results[use_batch_norm] = (train_accuracies, test_accuracies)

    return results


def plot_results(
    epochs: Sequence[int],
    results: Mapping[bool, tuple[Sequence[float], Sequence[float]]],
) -> None:
    import matplotlib.pyplot as plt

    train_with_batch_norm, test_with_batch_norm = results[True]
    train_without_batch_norm, test_without_batch_norm = results[False]
    plt.plot(epochs, train_with_batch_norm, marker="o", label="train with BatchNorm")
    plt.plot(epochs, test_with_batch_norm, marker="o", label="test with BatchNorm")
    plt.plot(epochs, train_without_batch_norm, marker="o", label="train without BatchNorm")
    plt.plot(epochs, test_without_batch_norm, marker="o", label="test without BatchNorm")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    plt.show()


def main() -> None:
    results = compare_epoch_accuracy()
    train_with_batch_norm, test_with_batch_norm = results[True]
    train_without_batch_norm, test_without_batch_norm = results[False]

    print("\nEpoch | BN train | BN test | No-BN train | No-BN test")
    for epoch, bn_train, bn_test, no_bn_train, no_bn_test in zip(
        EPOCHS,
        train_with_batch_norm,
        test_with_batch_norm,
        train_without_batch_norm,
        test_without_batch_norm,
        strict=True,
    ):
        print(f"{epoch:>5} | {bn_train:>8.4f} | {bn_test:>7.4f} | {no_bn_train:>11.4f} | {no_bn_test:>10.4f}")

    plot_results(EPOCHS, results)


if __name__ == "__main__":
    main()
