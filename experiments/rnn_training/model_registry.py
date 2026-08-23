"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: July 2026
"""
from deep_learning_pipeline.models import LSTMNetwork, GRUNetwork, VanillaRNNNetwork
from deep_learning_pipeline.configs import VanillaRNNCfg, LSTMCfg, GRUCfg

MODEL_REGISTRY = {

    "vanilla_rnn" : {
        "model" : VanillaRNNNetwork,
        "config" : VanillaRNNCfg
    },

    "gru": {
        "model" : GRUNetwork,
        "config" : GRUCfg
    },

    "lstm" : {
        "model" : LSTMNetwork,
        "config" : LSTMCfg
    }
}