hidden_layers = [1, 2, 3, 4, 5]
import time

import matplotlib.pyplot as plt

from experiments.mnist.mnist_NLNet_backprop import mnist

hidden_sizes = [[64], [64, 64], [64, 64, 64], [64, 64, 64, 64], [64, 64, 64, 64, 64]]
trainlist = []
testlist = []

for i in range(len(hidden_layers)):
    print("-----------------------")
    print(f"{i + 1}/{len(hidden_layers)}")
    print("-----------------------")
    time.sleep(0.05)
    (train_acc, test_acc) = mnist(hidden_layers[i], hidden_sizes[i])
    trainlist.append(train_acc)
    testlist.append(test_acc)

plt.plot(hidden_layers, trainlist, label="train")
plt.plot(hidden_layers, testlist, label="test")
plt.xlabel("hidden layers")
plt.ylabel("accuracy")
plt.legend()
plt.show()
