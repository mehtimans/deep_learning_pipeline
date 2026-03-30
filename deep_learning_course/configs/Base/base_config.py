"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: March 2026
"""

class BaseConfig():
    seed = 42
    device = 'cuda'
    class training:
        hidden_dims = [64, 32, 16]
        epochs = 3000
        batch_size = 64
        learning_rate = 1e-4

        activation = 'relu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid 
        optimizer = "adam" # e.g. "adam", "sgd"
        weight_decay = 1e-4 # L2 regularization, This extra term penalizes large weights.
        momentum = 0.9 # used when optimizer == "sgd"
        loss = "mse"  # e.g. "mse", "smooth_l1", "l1"
        loss_beta = 1.0 # used for SmoothL1Loss

        early_stopping = True
        patience = 100
        
        add_noise = True
        class noise:
            noise_std = 0.005
            noise_frac = 0.02

    class evaluation:
        val_split = 0.2
        load_run = -1

    class logger:
        train_label = 'MLP'
        log_dir = -1 

        
