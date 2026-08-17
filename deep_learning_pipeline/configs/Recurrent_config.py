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

class RecurrentCfg(BaseConfig):
    seed = 42
    device = "cuda"

    class data():
        vocab_size = 5000

    class training(BaseConfig.training):
        epochs = 30
        batch_size = 128
        learning_rate = 1e-3
        activation = 'relu'
        optimizer = "adam"
        weight_decay = 1e-4
        loss = "bce_logits"

        class trainer(BaseConfig.training.trainer):
            metrics = ["accuracy"]
            monitor = "accuracy"
            mode = "max"

            enable_plots = True
            early_stopping = False
            patience = 10

    class evaluation(BaseConfig.evaluation):
        val_split = 0.1
        test_split = 0.1
        load_run = -1

    class logger(BaseConfig.logger):
        experiment = "rnn_training"
        dataset = "rt_polarity"