"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: March 2026
"""

from deep_learning_course.configs import BaseConfig

class AutoEncoderCfg(BaseConfig):
    seed = 42 # Random seed for reproducibility
    device = 'cuda'
    class training(BaseConfig.training):
        encoder_hidden_dims = [128] # Dimensions of encoder hidden layers
        decoder_hidden_dims = [128] # Dimensions of decoder hidden layers
        classifier_hidden_dims = [16] # Dimensions of classifier hidden layers
        classifier_num_outputs = 10
        latent_size = 32
        epochs = 30
        autoencoder_batch_size = 256 # autoencoder Mini-batch size for training
        classifier_batch_size = 512 # classifier Mini-batch size for training
        learning_rate = 1e-3

        activation = 'relu' # Activation functions:  elu, relu, selu, crelu, lrelu, tanh, sigmoid
        optimizer = "adam" # Optimizer algorithm: "adam", "sgd","adamw"
        weight_decay = 1e-5 # L2 regularization, This extra term penalizes large weights.
        autoencoder_loss = "mse"  # e.g. "mse", "smooth_l1", "l1", "adamw", "crossentropy"
        classifier_loss = "crossentropy"  # e.g. "mse", "smooth_l1", "l1", "adamw", "crossentropy"

        class trainer(BaseConfig.training.trainer):
            autoencoder_trainer_name = "Autoencoder" # Identifier for this specific trainer run
            classifier_trainer_name = "Classifier" # Identifier for this specific trainer run
            autoencoder_metrics = ["mse", "rmse", "r2"] # Metrics to compute and track during training: "mse", "rmse", "r2", "accuracy"
            autoencoder_monitor = "rmse"  # Metric to monitor for early stopping and best model
            autoencoder_mode = "min" # Whether to minimize ('min') or maximize ('max') the monitored metric
            classifier_metrics = ["accuracy"] # Metrics to compute and track during training: "mse", "rmse", "r2", "accuracy"
            classifier_monitor = "accuracy"  # Metric to monitor for early stopping and best model
            classifier_mode = "max" # Whether to minimize ('min') or maximize ('max') the monitored metric
            enable_plots = True # save plots of metrics
            early_stopping = True
            autoencoder_patience = 5 # Number of epochs with no improvement before stopping
            classifier_patience = 25 # Number of epochs with no improvement before stopping

            add_noise = True # Noise injection for training data
            class noise(BaseConfig.training.trainer.noise):
                autoencoder_noise_std = 0.005 # Standard deviation of the Gaussian noise to add
                autoencoder_noise_frac = 0.02 # Fraction of samples that will receive noise perturbation
                classifier_noise_std = 0.0 # Standard deviation of the Gaussian noise to add
                classifier_noise_frac = 0.0 # Fraction of samples that will receive noise perturbation

    class evaluation(BaseConfig.evaluation):
        load_run = -1
        load_autoencoder = -1 # "/home/mehtimans/deep_learning_course/logs/Autoencoder/2026-04-01_01-37-32_/JIT_model.pt"

    class logger(BaseConfig.logger):
        train_label = 'Autoencoder'
        autoencoder_save_model_label = "Autoencoder_JIT_model"
        classifier_save_model_label = "Classifier_JIT_model"
        log_dir = -1 

        experiment = "HW1_Q1" # Name of the overall experiment
        dataset = "MNIST" # Name of the dataset being used
       