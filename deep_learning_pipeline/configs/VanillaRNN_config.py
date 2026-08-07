"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: July 2026
"""

from deep_learning_pipeline.configs import RecurrentCfg

class VanillaRNNCfg(RecurrentCfg):

    class training(RecurrentCfg.training):
        hidden_dims = [16, 32] # Dimensions of hidden layers

        class trainer(RecurrentCfg.training.trainer):
            trainer_name = "VanillaRNN" # Identifier for this specific trainer run

    class logger(RecurrentCfg.logger):
        train_label = "VanillaRNN"
        save_model_label = "VanillaRNN_JIT_model"
        log_dir = -1 # Directory to save logs (-1 means use default location)
