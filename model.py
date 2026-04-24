import torch
import torch.nn as nn
import torch.nn.functional as F


#Prunable Conv Layer
class PrunableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.gate_scores = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )

        self.padding = padding

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates

        return F.conv2d(x, pruned_weights, self.bias, padding=self.padding)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)


#Prunable Linear Layer
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)


#Prunable CNN Model
class PrunableCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = PrunableConv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = PrunableLinear(16 * 16 * 16, 64)
        self.fc2 = PrunableLinear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x) 

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_all_gates(self):
        gates = []
        for module in self.modules():
            if isinstance(module, (PrunableConv2d, PrunableLinear)):
                gates.append(module.get_gates().view(-1))
        return torch.cat(gates)