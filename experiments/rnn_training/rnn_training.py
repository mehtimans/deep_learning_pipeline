"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: July 2026
"""


import pandas as pd 
import os
import json 
from typing import Tuple, List

import torch 
from torch.utils.data import DataLoader

from model_registry import MODEL_REGISTRY

from deep_learning_pipeline.configs import RecurrentCfg, LSTMCfg, GRUCfg, VanillaRNNCfg
from deep_learning_pipeline.utils import get_args, set_seed, update_cfg_from_args, class_to_dict
from deep_learning_pipeline.utils import split_dataset, get_dataloader, compute_normalization_stats, get_log_dir, save_model_jit
from deep_learning_pipeline import DEEP_LEARNING_PIPELINE_RESOURCES_DIR
from text_processor import load_data, stratified_split_dataset, whitespace_tokenizer
from text_processor import TextDataset, collate_fn, Vocabulary

if __name__ == "__main__":

    # get args
    args = get_args()

    # load base configuration
    base_cfg = update_cfg_from_args(args, RecurrentCfg())

    # set the whole Experiment seed (be careful about this)
    base_cfg.seed = set_seed(base_cfg.seed)

    # load dataset
    folder_path = os.path.join(DEEP_LEARNING_PIPELINE_RESOURCES_DIR, "data", "rt_polarity")
    path_pos = os.path.join(folder_path, "rt-polarity.pos")
    path_neg = os.path.join(folder_path, "rt-polarity.neg")
    X, Y = load_data(path_pos, path_neg)

    # Tokenize the input texts using the whitespace tokenizer
    tokenized_inputs = [whitespace_tokenizer(sentence) for sentence in X]   

    X_train, Y_train, X_val, Y_val, X_test, Y_test = stratified_split_dataset(
        tokenized_inputs,
        Y,
        base_cfg.evaluation.val_split,
        base_cfg.evaluation.test_split,)

    vocab = Vocabulary(vocab_size=base_cfg.data.vocab_size)

    # Build vocabulary only from training samples
    vocab.build(X_train)

    # Encode each split using the same vocabulary
    X_train = vocab.encode_batch(X_train)
    X_val = vocab.encode_batch(X_val)
    X_test = vocab.encode_batch(X_test)

    train_ds = TextDataset(X_train, Y_train)
    val_ds = TextDataset(X_val, Y_val)
    test_ds = TextDataset(X_test, Y_test)

    train_dl = DataLoader(
        train_ds,
        batch_size=base_cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn)

    val_dl = DataLoader(
        val_ds,
        batch_size=base_cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn)

    test_dl = DataLoader(
        test_ds,
        batch_size=base_cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn)

    for name, experiment in MODEL_REGISTRY.items():

        # final experiment configuration 
        cfg = update_cfg_from_args(args, experiment["config"]())
        cfg_dict = class_to_dict(cfg)
        print(f"{name} configuration initialized")
        print(json.dumps(cfg_dict, indent=4))

        # set seed 
        cfg.seed = set_seed(cfg.seed)

        # get logging directory  
        cfg.logger.log_dir = get_log_dir(cfg.logger.log_dir, cfg.logger.train_label)

