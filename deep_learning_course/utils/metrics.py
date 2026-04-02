"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: April 2026
"""

from torch import Tensor
from typing import Callable

def mse(preds: Tensor, targets: Tensor) -> Tensor:
    """
    Compute mean squared error between predictions and targets

    """
    return ((preds - targets) ** 2).mean()

def rmse(preds: Tensor, targets: Tensor) -> Tensor:
    """
    Compute root mean squared error (sqrt of MSE)
    
    """
    return ((preds - targets) ** 2).mean().sqrt()

def r2(preds: Tensor, targets: Tensor) -> Tensor:
    """
    Compute R² score (coefficient of determination)
    
    """
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    if ss_tot == 0:
        return Tensor([0.0])
    return (1 - ss_res / ss_tot)

def accuracy(preds: Tensor, targets: Tensor) -> Tensor:
    """
    Compute R² score (coefficient of determination)
    
    """
    preds_ = preds.argmax(dim=1)
    return (preds_ == targets).float().mean()

METRIC_REGISTRY: dict[str, callable] = {
    "mse": mse,
    "rmse": rmse,
    "r2": r2,
    "accuracy": accuracy
}
