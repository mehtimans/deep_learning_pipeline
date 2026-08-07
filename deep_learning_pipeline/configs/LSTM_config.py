"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: July 2026
"""

from deep_learning_pipeline.configs import BaseConfig

class LSTMCfg(BaseConfig):
    seed = 42 # Random seed for reproducibility
    device = 'cuda'
    class training(BaseConfig.training):
        hidden_dims = [16, 32] # Dimensions of hidden layers
        epochs = 30
        batch_size = 128 # Mini-batch size for training
        learning_rate = 1e-3

        activation = 'relu' # Activation functions:  elu, relu, selu, crelu, lrelu, tanh, sigmoid
        optimizer = "adam" # Optimizer algorithm: "adam", "sgd","adamw"
        weight_decay = 1e-4 # L2 regularization, This extra term penalizes large weights.
        loss = "mse"  # Loss function to minimize: "mse", "smooth_l1", "l1", "crossentropy"
        
        class trainer(BaseConfig.training.trainer):
            trainer_name = "LSTM" # Identifier for this specific trainer run
            metrics = ["mse", "rmse", "r2"] # Metrics to compute and track during training: "mse", "rmse", "r2", "accuracy"
            monitor = "rmse"  # Metric to monitor for early stopping and best model
            mode = "min" # Whether to minimize ('min') or maximize ('max') the monitored metric
            enable_plots = True # save plots of metrics
            early_stopping = False
            patience = 100 # Number of epochs with no improvement before stopping
            
            add_noise = True # Noise injection for training data
            class noise(BaseConfig.training.trainer.noise):
                noise_std = 0.005 # Standard deviation of the Gaussian noise to add
                noise_frac = 0.02 # Fraction of samples that will receive noise perturbation

    class evaluation(BaseConfig.evaluation):
        val_split = 0.2 # Proportion of training data to use for validation
        load_run = -1

    class logger(BaseConfig.logger):
        train_label = "LSTM"
        save_model_label = "LSTM_JIT_model"
        log_dir = -1 # Directory to save logs (-1 means use default location)

        experiment = "rnn_training" # Name of the overall experiment
        dataset = "rt_polarity" # Name of the dataset being used


