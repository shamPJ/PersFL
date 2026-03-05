import torch
from torch import nn

class LinReg(nn.Module):
    """
    Linear regression with d parameters. No bias by default.
    """
    def __init__(self, n_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)

    def forward(self, x):
        return self.linear(x)