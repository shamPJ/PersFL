import torch
from torch import nn

def MSE(pred, target, model=None, extra=None):
    return torch.mean((pred - target) ** 2)
def MSE_params(pred, target, model=None, extra=None):
    # we expect linear model here
    return torch.mean((pred - target) ** 2)

def accuracy(pred, target, model=None, extra=None):
    return (pred.argmax(dim=1) == target).float().mean()
def F1(pred, target, model=None, extra=None):
    # Placeholder for proper F1 computation
    return torch.tensor(0.0)