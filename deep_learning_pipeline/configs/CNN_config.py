"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: April 2026
"""

from deep_learning_pipeline.configs import BaseConfig

class CNNCfg(BaseConfig):
    seed = -1 # Random seed for reproducibility
    device = 'cuda'
    class training(BaseConfig.training):
        mlp_hidden_dims = [512] # Dimensions of hidden layers
        epochs = 150
        batch_size = 512 # Mini-batch size for training
        learning_rate = 1e-4

        mlp_activation = 'relu' # Activation functions:  elu, relu, selu, crelu, lrelu, tanh, sigmoid
        cnn_activation = 'relu' # Activation functions:  elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
        optimizer = "adam" # Optimizer algorithm: "adam", "sgd","adamw"
        weight_decay = 1e-4 # L2 regularization, This extra term penalizes large weights.
        loss = "crossentropy"  # Loss function to minimize: "mse", "smooth_l1", "l1", "crossentropy"
        
        # Convolutional blocks defining the CNN architecture.
        # Each block contains:
        # - out_channels: # feature maps
        # - num_convs: # sequential Conv2D layers
        # - kernel_size, stride, padding: conv parameters
        # - pool: optional MaxPool config or None
        # - batch_normalization: enable BatchNorm2d
        conv_blocks = [
            {
                "out_channels": 32,
                "num_convs": 2,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "pool": {"size": 2, "stride": 1, "padding": 0},
                "batch_normalization": True
            },
            {
                "out_channels": 64,
                "num_convs": 2,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "pool": {"size": 2, "stride": 2, "padding": 0},
                "batch_normalization": True
            },
            {
                "out_channels": 128,
                "num_convs": 2,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "pool": {"size": 2, "stride": 2, "padding": 0},
                "batch_normalization": True
            },
        ]

        class trainer(BaseConfig.training.trainer):
            trainer_name = "CNN" # Identifier for this specific trainer run
            metrics = ["accuracy"] # Metrics to compute and track during training: "mse", "rmse", "r2", "accuracy"
            monitor = "accuracy"  # Metric to monitor for early stopping and best model
            mode = "max" # Whether to minimize ('min') or maximize ('max') the monitored metric
            enable_plots = True # save plots of metrics
            early_stopping = True
            patience = 15 # Number of epochs with no improvement before stopping
            
            add_noise = True # Noise injection for training data
            class noise(BaseConfig.training.trainer.noise):
                noise_std = 0.000 # Standard deviation of the Gaussian noise to add
                noise_frac = 0.00 # Fraction of samples that will receive noise perturbation
        
    class datageneration:
        num_classes = 10
        num_channels = 1
        samples_per_class = 2000
        image_dim = 14
        mean_range = [-5, 4]
        variance = 2.25

    class evaluation(BaseConfig.evaluation):
        val_split = 0.2 # Proportion of training data to use for validation
        load_run = -1

    class logger(BaseConfig.logger):
        train_label = 'CNN'
        save_model_label = "CNN_JIT_model"
        log_dir = -1 # Directory to save logs (-1 means use default location)

        experiment = "cnn_training" # Name of the overall experiment
        dataset = "Toy_Data" # Name of the dataset being used

