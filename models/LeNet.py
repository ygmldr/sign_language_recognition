import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * (5 * 5), 120)
        self.fc2 = nn.Linear(120, 84)
        # it's ok to use 26 channels or 24 channels
        # in convenience, we use 26 channels
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (1, 28, 28) => (16, 28, 28)
        x = self.pool1(x)  # (16, 28, 28) => (16, 14, 14)

        x = F.relu(self.conv2(x))  # (16, 14, 14) => (32, 10, 10)
        x = self.pool2(x)  # (32, 10, 10) => (32, 5, 5)

        x = x.view(-1, 32 * (5 * 5))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    print(LeNet())
