"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: March 2026
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import json 

import torch 

from deep_learning_course.models import MLPNetwork
from deep_learning_course.configs import HW1Q2cfg
from deep_learning_course.utils import get_args, set_seed, update_cfg_from_args, class_to_dict
from deep_learning_course.utils import split_dataset, compute_normalization_stats, get_log_dir, save_model_jit
from deep_learning_course.utils import get_loss, get_optimizer
from deep_learning_course.pipeline import Trainer
from deep_learning_course import DEEP_LEARNING_COURSE_RESOURCES_DIR, DEEP_LEARNING_COURSE_ROOT_DIR


def load_dataset(data_dir: str):
    """
    Load and preprocess the Life Expectancy dataset.

    csv_path : str
        Full path to the CSV file.

    Returns
    Feature : torch.Tensor
        Feature matrix (float32).
    Target : torch.Tensor
        Target values (Life Expectancy).
    """

    df = pd.read_csv(data_dir, sep=",") # Read CSV

    # Preprocessing dataset 
    df.columns = df.columns.str.strip().str.replace(" ","_") # Clean column names
    df = df.drop(columns=["Country"]) # remove country column 
    df["Status"] = df["Status"].map({"Developed": 1, "Developing": 0}) # Encode categorical column
    # print(df["Status"].unique())

    # Handle missing values with median
    categorical_cols = ["Status"]
    numeric_cols = df.columns.drop(categorical_cols)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df["Status"] = df["Status"].fillna(df["Status"].mode()[0])
    # print(df.isna().sum())

    # Convert to torch tensors
    Features = torch.tensor(df.drop(columns=["Life_expectancy"]).values, dtype=torch.float32)
    Targets = torch.tensor(df[["Life_expectancy"]].values, dtype=torch.float32)
    
    return Features, Targets



if __name__ == "__main__":
    
    # get args
    args = get_args()
    
    # final configuration 
    cfg = update_cfg_from_args(args, HW1Q2cfg())
    cfg_dict = class_to_dict(cfg)
    print(json.dumps(cfg_dict, indent=4))

    # set seed 
    cfg.seed = set_seed(cfg.seed)

    # get logging directory  
    cfg.logger.log_dir = get_log_dir(cfg.logger.log_dir, cfg.logger.train_label)

    # load dataset 
    csv_folder = os.path.join(DEEP_LEARNING_COURSE_RESOURCES_DIR, "data", "HW1","HW1_Q2_dataset")
    csv_path = os.path.join(csv_folder, "Life Expectancy Data.csv")
    X, Y = load_dataset(csv_path)
    print("X shape", X.size())
    print("Y shape", Y.size())

    # split
    train_ds, val_ds, train_dl, val_dl = split_dataset(X, Y, cfg.training.batch_size, cfg.evaluation.val_split)

    # compute_normalization_stats
    mean, std = compute_normalization_stats(train_ds)
    print("Feature mean:", mean.cpu().numpy())
    print("Feature std: ", std.cpu().numpy())
    print(f"output min: {Y.min().item():.4f}, max: {Y.max().item():.4f}, std: {Y.std().item():.4f}")

    # load model
    mlp_model = MLPNetwork(
                    num_inputs=X.size(1), 
                    num_outputs=Y.size(1),
                    network_hidden_dims = cfg.training.hidden_dims,
                    activation = cfg.training.activation)
    
    mlp_model.set_normalization(mean, std)  

    # loss function 
    loss_fn = get_loss(cfg.training.loss, reduction="mean")

    # Optimizer 
    optimizer = get_optimizer(cfg.training.optimizer, mlp_model.parameters(), 
                              lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    
    # Train the model 
    trainer = Trainer(
        mlp_model,
        train_dl,
        val_dl,
        optimizer,
        loss_fn,
        epochs=cfg.training.epochs,
        device=cfg.device,
        noise_std=cfg.training.noise.noise_std,
        noise_frac=cfg.training.noise.noise_frac,
        early_stopping=cfg.training.early_stopping,
        patience=cfg.training.patience,
        log_dir=cfg.logger.log_dir 
    )

    best_model = trainer.train() # Note: best_model is stored on CPU for portability

    # save model as a jit file
    model_path = save_model_jit(best_model, cfg.logger.log_dir)

    # save config
    config_path = os.path.join(cfg.logger.log_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=4)
    
    print(f"config saved to {config_path}")


