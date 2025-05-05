import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out
