"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: March 2026
"""
from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
from datetime import datetime
import os
import copy

from deep_learning_course import DEEP_LEARNING_COURSE_ROOT_DIR

def split_dataset(
    X: Tensor,
    Y: Tensor,
    batch_size: int = 32,
    val_split: float = 0.2,
) -> Tuple[Subset, Subset, DataLoader, DataLoader]:
    """
    Split tensors into training and validation sets and create DataLoaders.

    Args:
        X (Tensor): Feature tensor of shape (N, D)
        Y (Tensor): Target tensor of shape (N, 1) or (N,)
        batch_size (int): Batch size for the DataLoaders
        val_split (float): Fraction of the dataset used for validation

    Returns:
        train_dataset (Subset): Training subset
        val_dataset (Subset): Validation subset
        train_loader (DataLoader): DataLoader for training
        val_loader (DataLoader): DataLoader for validation
    """

    ds = TensorDataset(X, Y)

    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val

    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_ds, val_ds, train_dl, val_dl


def compute_normalization_stats(train_ds: Subset) -> Tuple[Tensor, Tensor]:
    """
    Compute feature-wise mean and standard deviation using ONLY the training data.

    Args:
        train_dataset (Subset): Training subset returned by random_split

    Returns:
        mean (Tensor): Feature-wise mean of shape (D,)
        std (Tensor): Feature-wise standard deviation of shape (D,)
    """
    X = train_ds.dataset.tensors[0]   # feature tensor
    train_indices = train_ds.indices
    train_X = X[train_indices]

    mean = train_X.mean(dim=0)
    std = train_X.std(dim=0)
    std = torch.clamp(std, min=1e-8)

    return mean, std

def save_model_jit(model, log_dir: str)->str:
    """
    Save a PyTorch model as a TorchScript (JIT) file.
    
    """
    
    os.makedirs(log_dir, exist_ok=True)

    model = copy.deepcopy(model).to("cpu").eval()
    model_path = os.path.join(log_dir, 'JIT_model.pt')
    
    # save as jit
    try:
        scripted = torch.jit.script(model)
        scripted.save(model_path)
        print(f"Model saved to {model_path}")

    except Exception as e:
        raise RuntimeError(f"Error scripting model for export. ") from e
    
    return model_path

def get_log_dir (log_dir, train_label: str)-> str: 
    """
    Create or resolve the logging directory for a training run.

    If log_dir == -1, a timestamped directory is automatically created:
    logs/train_label/timestamp/
    
    """

    if log_dir == -1:
        # directory to save
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
        log_root = os.path.join(DEEP_LEARNING_COURSE_ROOT_DIR, "logs")
        log_dir = os.path.join(log_root, train_label, timestamp)
        os.makedirs(log_dir, exist_ok=True)

    try:
        # check if path exists as a file
        if os.path.isfile(log_dir):
            raise ValueError(f"{log_dir} exists and is a file, expected a directory.")

        os.makedirs(log_dir, exist_ok=True)

    except OSError as e:
        raise RuntimeError(f"Could not create or access log directory: {log_dir}") from e

    return log_dir