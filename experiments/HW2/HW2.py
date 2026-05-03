"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: April 2026
"""

import pandas as pd 
import os
import json 
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

import torch 
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D

from deep_learning_course.models import CNNNetwork, CNNNetworkIAM
from deep_learning_course.configs import HW2cfg
from deep_learning_course.utils import get_args, set_seed, update_cfg_from_args, class_to_dict
from deep_learning_course.utils import split_dataset, get_dataloader, get_log_dir, save_model_jit, count_trainable_params
from deep_learning_course.utils import get_loss, get_optimizer, pca_torch
from deep_learning_course.pipeline import Trainer
from deep_learning_course import DEEP_LEARNING_COURSE_RESOURCES_DIR, DEEP_LEARNING_COURSE_ROOT_DIR

from HW2_fun import Dataset_generation, compute_normalization_stats, compute_covariance_matrix, plot_mean_histograms
from HW2_fun import mean_classifier_confusion_matrix, evaluate_model, plot_PCA, plot_PCA_3D, cosine_test
from HW2_fun import evaluate_four_metrics, plot_class_center_angles


if __name__ == "__main__":
    
    # get args
    args = get_args()
    
    # final configuration 
    cfg = update_cfg_from_args(args, HW2cfg())
    cfg_dict = class_to_dict(cfg)
    print(json.dumps(cfg_dict, indent=4))

    # set seed 
    cfg.seed = set_seed(cfg.seed)

    # get logging directory  
    cfg.logger.log_dir = get_log_dir(cfg.logger.log_dir, cfg.logger.train_label)

    # load and split dataset 
    train_ds, val_ds = Dataset_generation(cfg.datageneration, cfg.evaluation.val_split)
    print("Train dataset size:", len(train_ds))
    print("Val dataset size:", len(val_ds))    
    # x, y = train_ds[0] 
    # print(x.size(), y, type(y))  
    # print(type(train_ds))

    # compute_covariance_matrix(train_ds)
    # plot_mean_histograms(train_ds)
    # mean_classifier_confusion_matrix(train_ds, val_ds)
    
    # get dataloaders
    train_dl, val_dl = get_dataloader(train_ds, val_ds, cfg.training.batch_size)
    print("Number of training batches:", len(train_dl))
    x, _ = next(iter(train_dl))
    _, data_channel, data_height, data_width = x.size()
    print("Data Channel:", data_channel)
    print("Data Height:", data_height)
    print("Data width:", data_width)

    # compute_normalization_stats
    mean, std = compute_normalization_stats(train_ds)
    print("Feature mean:", mean.cpu().numpy())
    print("Feature std: ", std.cpu().numpy())

    # load model
    CNN_model = CNNNetwork(
        input_channels=data_channel,
        input_height=data_height,
        input_width=data_width,
        num_outputs=cfg.datageneration.num_classes,
        conv_blocks=cfg.training.conv_blocks,
        cnn_activation=cfg.training.cnn_activation,
        mlp_network_hidden_dims=cfg.training.mlp_hidden_dims,
        mlp_activation=cfg.training.mlp_activation
    ).to(cfg.device)

    CNN_num_params = count_trainable_params(CNN_model)
    print(f"CNN Network has {CNN_num_params:,} trainable parameters")

    CNN_model.set_normalization(mean, std)  

    # loss function 
    loss_fn = get_loss(cfg.training.loss, reduction="mean")

    # Optimizer 
    optimizer = get_optimizer(cfg.training.optimizer, CNN_model.parameters(), 
                              lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    
    # # Scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",          # because you monitor RMSE
    #     factor=0.5,          # LR *= factor when plateau
    #     patience=80,         # epochs with no improvement before reducing LR
    #     min_lr=1e-4,
    # )
    
    # Train the model 
    trainer = Trainer(
        CNN_model,
        train_dl,
        val_dl,
        optimizer,
        loss_fn,
        trainer_name=cfg.training.trainer.trainer_name,
        epochs=cfg.training.epochs,
        # scheduler=scheduler, 
        device=cfg.device,
        noise_std=cfg.training.trainer.noise.noise_std,
        noise_frac=cfg.training.trainer.noise.noise_frac,
        metrics=cfg.training.trainer.metrics,
        monitor=cfg.training.trainer.monitor,
        mode=cfg.training.trainer.mode,
        early_stopping=cfg.training.trainer.early_stopping,
        patience=cfg.training.trainer.patience,
        log_dir=cfg.logger.log_dir 
    )

    best_model = trainer.train() # Note: best_model is stored on CPU for portability

    # save model as a jit file
    model_path = save_model_jit(best_model, cfg.logger.log_dir, cfg.logger.save_model_label)

    # evaluate_model(CNN_model, val_dl)
    # plot_PCA(CNN_model, val_dl)
    # plot_PCA_3D(CNN_model, val_dl)
    # evaluate_four_metrics(CNN_model, val_dl)
    plot_class_center_angles(CNN_model, val_dl)
    # cosine_test()

    ## Training second model with IAM Loss
    # load model
    CNN_model_IAM = CNNNetworkIAM(
        input_channels=data_channel,
        input_height=data_height,
        input_width=data_width,
        num_outputs=cfg.datageneration.num_classes,
        conv_blocks=cfg.training.conv_blocks,
        cnn_activation=cfg.training.cnn_activation,
        mlp_network_hidden_dims=cfg.training.mlp_hidden_dims,
        mlp_activation=cfg.training.mlp_activation
    ).to(cfg.device)
    
    CNN_IAM_num_params = count_trainable_params(CNN_model_IAM)
    print(f"CNN Network has {CNN_IAM_num_params:,} trainable parameters")

    CNN_model_IAM.set_normalization(mean, std)  

    # IAM loss function 
    cfg.training.loss = "angularmargin"
    beta=0.3
    # s=30.0
    # m=0.3
    loss_fn = get_loss(cfg.training.loss, s=30.0, m=0.5, reduction="mean")

    # Optimizer 
    optimizer = get_optimizer(cfg.training.optimizer, CNN_model_IAM.parameters(), 
                              lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    
    # # Scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",          # because you monitor RMSE
    #     factor=0.5,          # LR *= factor when plateau
    #     patience=80,         # epochs with no improvement before reducing LR
    #     min_lr=1e-4,
    # )
    
    # Train the model 
    cfg.training.trainer.trainer_name = "CNN_WITH_IAM"
    trainer_IAM = Trainer(
        CNN_model_IAM,
        train_dl,
        val_dl,
        optimizer,
        loss_fn,
        trainer_name=cfg.training.trainer.trainer_name,
        epochs=cfg.training.epochs,
        # scheduler=scheduler, 
        device=cfg.device,
        noise_std=cfg.training.trainer.noise.noise_std,
        noise_frac=cfg.training.trainer.noise.noise_frac,
        metrics=cfg.training.trainer.metrics,
        monitor=cfg.training.trainer.monitor,
        mode=cfg.training.trainer.mode,
        early_stopping=cfg.training.trainer.early_stopping,
        patience=5,
        log_dir=cfg.logger.log_dir 
    )

    best_model_IAM = trainer_IAM.train() # Note: best_model is stored on CPU for portability

    # save model as a jit file
    cfg.logger.save_model_label = "CNN_IAM_JIT_model"
    model_path = save_model_jit(best_model_IAM, cfg.logger.log_dir, cfg.logger.save_model_label)

    # evaluate_model(CNN_model_IAM, val_dl)
    # plot_PCA(CNN_model_IAM, val_dl)
    # plot_PCA_3D(CNN_model_IAM, val_dl)
    # evaluate_four_metrics(CNN_model_IAM, val_dl)
    plot_class_center_angles(CNN_model_IAM, val_dl)

    # save config
    config_path = os.path.join(cfg.logger.log_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=4)
    print(f"config saved to {config_path}")





