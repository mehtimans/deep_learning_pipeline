"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: March 2026
"""

import torch
from torch import Tensor
import torch.nn as nn
from deep_learning_course.utils import IAMLoss
from deep_learning_course.utils import AngularMarginLoss

def get_activation(name: str) -> nn.Module:
    '''
    Map a string name to an activation module instance.
    
    '''
    name = name.lower()
    if name == "elu": return nn.ELU()
    if name == "relu": return nn.ReLU()
    if name == "lrelu": return nn.LeakyReLU()
    if name == "tanh": return nn.Tanh()
    if name == "sigmoid": return nn.Sigmoid()
    if name == "softsign": return nn.Softsign()
    raise ValueError(f"Unknown activation '{name}'")

def get_loss(name: str, **kwargs) -> nn.Module:
    '''
    Map a string name to a PyTorch loss module.
    Extra keyword args (in kwargs) are forwarded to the loss constructor.
    '''
    name = name.lower()
    if name == "mse": return nn.MSELoss(**kwargs)
    if name == "smooth_l1": return nn.SmoothL1Loss(**kwargs)
    if name == "l1": return nn.L1Loss(**kwargs)
    if name == "crossentropy": return nn.CrossEntropyLoss(**kwargs)
    if name == "iam": return IAMLoss(**kwargs)
    if name == "angularmargin": return AngularMarginLoss(**kwargs)
    raise ValueError(f"Unknown loss '{name}'")

def get_optimizer(name: str, params, lr: float, **kwargs) -> torch.optim.Optimizer:
    """
    Build an optimizer from its string name.

    Args:
        name: Optimizer name ("sgd", "adam", "adamw", ...), case-insensitive.
        params: Iterable of model parameters (e.g. model.parameters()).
        lr: Base learning rate.
        **kwargs: Extra optimizer-specific keyword arguments, e.g.
                  momentum=0.9 for SGD, weight_decay=1e-4, betas=(0.9, 0.999) for Adam, etc.

    Returns:
        A constructed torch.optim.Optimizer instance.
    """
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, **kwargs)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, **kwargs)

    raise ValueError(f"Unknown optimizer '{name}'")

def pca_torch(X: Tensor , n_components: int = 2) -> Tensor:
    """
    Perform PCA using Torch's SVD.
    Input:
      X: [n_samples, n_features]
    Output:
      X_pca: [n_samples, n_components]
    """
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean

    U, S, V = torch.linalg.svd(X_centered, full_matrices=False) 
    components = V[:, :n_components]

    X_pca = torch.matmul(X_centered, components)
    return X_pca
