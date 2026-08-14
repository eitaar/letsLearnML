import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from dataset.mnist import load_mnist


def to_torch_data(x: np.ndarray, t: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x_tensor = torch.from_numpy(x.astype(np.float32, copy=False))
    t_tensor = torch.from_numpy(t)
    if t_tensor.ndim == 2:
        t_tensor = t_tensor.argmax(dim=1)
    return x_tensor, t_tensor.to(dtype=torch.long)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    x_train, t_train = to_torch_data(x_train, t_train)
    x_test, t_test = to_torch_data(x_test, t_test)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(net.parameters(), lr=0.01)
    batch_size = 100
    epochs = 3
    steps_per_epoch = 600

    for epoch in range(epochs):
        permutation = torch.randperm(x_train.size(0))
        for step in range(steps_per_epoch):
            batch_indices = permutation[step * batch_size : (step + 1) * batch_size]
            x_batch = x_train[batch_indices]
            t_batch = t_train[batch_indices]

            optimizer.zero_grad()
            outputs = net(x_batch)
            loss = criterion(outputs, t_batch)
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch + 1}/{epochs}: loss={loss.item():.4f}")

    print("Finished Training")

    with torch.no_grad():
        outputs = net(x_test)
        predicted = outputs.argmax(dim=1)
        accuracy = (predicted == t_test).float().mean().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * accuracy:.2f} %")


if __name__ == "__main__":
    main()
