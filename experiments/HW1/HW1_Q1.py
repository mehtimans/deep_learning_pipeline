"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: March 2026
"""

import pandas as pd 
import os
import json 
from typing import Tuple

import torch 
from torchvision import datasets, transforms
from torch.utils.data import Dataset


from deep_learning_course.models import MLPNetwork
from deep_learning_course.configs import HW1Q1cfg
from deep_learning_course.utils import get_args, set_seed, update_cfg_from_args, class_to_dict
from deep_learning_course.utils import split_dataset, get_dataloader, compute_normalization_stats, get_log_dir, save_model_jit
from deep_learning_course.utils import get_loss, get_optimizer
from deep_learning_course.pipeline import Trainer
from deep_learning_course import DEEP_LEARNING_COURSE_RESOURCES_DIR, DEEP_LEARNING_COURSE_ROOT_DIR


def load_dataset(root_path: str)-> Tuple[Dataset, Dataset]:
    """
    Loads the MNIST dataset and applies transformations:
    - Converts PIL images to tensors (values in [0, 1])
    - Flattens each image into a 784-dimensional vector
    """

    # transform = transforms.PILToTensor() # without normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_ds = datasets.MNIST(
        root=root_path,
        train=True,
        download=False,
        transform=transform
    )

    val_ds = datasets.MNIST(
        root=root_path,
        train=False,
        download=False,
        transform=transform
    )
        
    return train_ds, val_ds

if __name__ == "__main__":

    # get args
    args = get_args()

    # final configuration 
    cfg = update_cfg_from_args(args, HW1Q1cfg())
    cfg_dict = class_to_dict(cfg)
    print(json.dumps(cfg_dict, indent=4))

    # set seed 
    cfg.seed = set_seed(cfg.seed)

    # get logging directory  
    cfg.logger.log_dir = get_log_dir(cfg.logger.log_dir, cfg.logger.train_label)

    # load dataset 
    data_root_path = os.path.join(DEEP_LEARNING_COURSE_RESOURCES_DIR, "data")
    train_ds, val_ds = load_dataset(data_root_path)
    print("Train size:", len(train_ds))
    print("Validation size:", len(val_ds))
    # labels = [y for _, y in val_ds]
    # x, y = train_ds[0] 
    # print(x.size(), type(y))     
    # print(labels)

    # get dataloaders
    train_dl, val_dl = get_dataloader(train_ds, val_ds, cfg.training.batch_size)
    # x, y = next(iter(train_dl))
    # print(x.size())