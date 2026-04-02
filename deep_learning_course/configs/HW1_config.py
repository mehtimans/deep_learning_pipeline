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

class HW1Q1cfg(BaseConfig):
    seed = 42
    device = 'cuda'
    class training(BaseConfig.training):
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [128, 256]
        classifier_hidden_dims = [4]
        classifier_num_outputs = 10
        latent_size = 32
        epochs = 2
        autoencoder_batch_size = 256
        classifier_batch_size = 512
        learning_rate = 1e-3

        activation = 'relu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid 
        optimizer = "adam" # e.g. "adam", "sgd"
        weight_decay = 1e-5 # L2 regularization, This extra term penalizes large weights.
        autoencoder_loss = "mse"  # e.g. "mse", "smooth_l1", "l1", "adamw", "crossentropy"
        classifier_loss = "crossentropy"  # e.g. "mse", "smooth_l1", "l1", "adamw", "crossentropy"


        early_stopping = True
        patience = 10
        
        add_noise = True
        class noise(BaseConfig.training.noise):
            noise_std = 0.01
            noise_frac = 0.02

    class evaluation(BaseConfig.evaluation):
        load_run = -1
        load_autoencoder = "/home/mehtimans/deep_learning_course/logs/Autoencoder/2026-04-01_01-37-32_/JIT_model.pt"

    class logger(BaseConfig.logger):
        train_label = 'Autoencoder'
        log_dir = -1 

        experiment = "HW1_Q1"
        dataset = "MNIST"
       


class HW1Q2cfg(BaseConfig):
    seed = 42
    device = 'cuda'
    class training(BaseConfig.training):
        hidden_dims = [64, 32]
        epochs = 3000
        batch_size = 64
        learning_rate = 1e-4

        activation = 'relu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid 
        optimizer = "adam" # e.g. "adam", "sgd"
        weight_decay = 1e-4 # L2 regularization, This extra term penalizes large weights.
        momentum = 0.9 # used when optimizer == "sgd"
        loss = "mse"  # e.g. "mse", "smooth_l1", "l1", "crossentropy"
        loss_beta = 1.0 # used for SmoothL1Loss

        early_stopping = True
        patience = 100
        
        add_noise = True
        class noise(BaseConfig.training.noise):
            noise_std = 0.005
            noise_frac = 0.02

    class evaluation(BaseConfig.evaluation):
        val_split = 0.2
        load_run = -1

    class logger(BaseConfig.logger):
        train_label = 'MLP'
        log_dir = -1 

        experiment = "HW1_Q2"
        dataset = "Life_Expectancy_Data"