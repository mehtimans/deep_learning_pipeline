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
    seed = 42 # Random seed for reproducibility
    device = 'cuda'

    class model: 
        hidden_dims = [64, 32, 16] # Dimensions of hidden layers
        activation = 'relu' # Activation functions:  elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class training:
        epochs = 3000
        batch_size = 64 # Mini-batch size for training
        learning_rate = 1e-4

        optimizer = "adam" # Optimizer algorithm: "adam", "sgd","adamw", "iam"
        weight_decay = 1e-4 # L2 regularization, This extra term penalizes large weights.
        momentum = 0.9 # Momentum term (used when optimizer is "sgd")
        loss = "mse"  # e.g. "mse", "smooth_l1", "l1", "crossentropy"
        loss_beta = 1.0 # used for SmoothL1Loss
        
        class trainer():
            trainer_name = "MLP" # Identifier for this specific trainer run
            metrics = ["mse", "rmse", "r2"] # Metrics to compute and track during training: "mse", "rmse", "r2", "accuracy"
            monitor = "rmse"  # Metric to monitor for early stopping and best model
            mode = "min" # Whether to minimize ('min') or maximize ('max') the monitored metric
            enable_plots = True # save plots of metrics
            early_stopping = True
            patience = 100 # Number of epochs with no improvement before stopping

            add_noise = True # Noise injection for training data
            class noise():
                noise_std = 0.005 # Standard deviation of the Gaussian noise to add
                noise_frac = 0.02 # Fraction of samples that will receive noise perturbation

    class evaluation:
        val_split = 0.2 # Proportion of training data to use for validation
        test_split = 0.15
        load_run = -1 

    class logger:
        train_label = 'MLP'
        save_model_label = "JIT_model"
        log_dir = -1 # Directory to save logs (-1 means use default location)

        experiment = "exp" # Name of the overall experiment
        dataset = "Dataset" # Name of the dataset being used
