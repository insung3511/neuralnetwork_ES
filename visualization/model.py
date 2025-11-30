import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 0):
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SimpleMLP(nn.Module):
    """Transformer-based model for 1D regression."""

    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.net = nn.Sequential(self.net1, self.net2)

    def forward(self, x):
        return self.net(x)


def make_dataset(n: int = 200):
    """Create a simple 1D regression dataset (x, y).

    Returns torch tensors of shape (n,1).
    """
    x = torch.linspace(-2.0, 2.0, n).unsqueeze(1)
    y = (0.5 * torch.sin(2.5 * x) + 0.3 * x).to(torch.float32)
    return x, y
