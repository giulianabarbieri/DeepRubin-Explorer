import torch
import torch.nn as nn

class LightCurveCNN1D(nn.Module):
    def __init__(self, n_channels=2, n_times=100, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
