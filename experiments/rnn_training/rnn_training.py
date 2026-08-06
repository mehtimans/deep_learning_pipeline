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

from model_registry import MODEL_REGISTRY
from deep_learning_pipeline.utils import get_args, set_seed, update_cfg_from_args, class_to_dict
from deep_learning_pipeline.utils import split_dataset, get_dataloader, compute_normalization_stats, get_log_dir, save_model_jit
from deep_learning_pipeline import DEEP_LEARNING_PIPELINE_RESOURCES_DIR
from text_processor import load_data, whitespace_tokenizer, build_vocabulary


if __name__ == "__main__":

    # get args
    args = get_args()

    # load dataset
    folder_path = os.path.join(DEEP_LEARNING_PIPELINE_RESOURCES_DIR, "data", "rt_polarity")
    path_pos = os.path.join(folder_path, "rt-polarity.pos")
    path_neg = os.path.join(folder_path, "rt-polarity.neg")
    X, Y = load_data(path_pos, path_neg)
    print(f"Total samples: {len(X)}")

    # whitespace tocenize the input text 
    tokenized_inputs = [
        whitespace_tokenizer(sentence)
        for sentence in X]

    # print(f"length of tokenized X is {len(tokenized_inputs)}")
    # print(f"tokenized X is {tokenized_inputs[:3]}")

    build_vocabulary(tokenized_inputs)

    # for name, experiment in MODEL_REGISTRY.items():

    #     # final experiment configuration 
    #     cfg = update_cfg_from_args(args, experiment["config"]())
    #     cfg_dict = class_to_dict(cfg)
    #     print(f"{name} configuration initialized")
    #     print(json.dumps(cfg_dict, indent=4))

    #     # set seed 
    #     cfg.seed = set_seed(cfg.seed)

    #     # get logging directory  
    #     cfg.logger.log_dir = get_log_dir(cfg.logger.log_dir, cfg.logger.train_label)

    #     # split
    #     train_ds, val_ds, test_ds = split_dataset(X, Y, cfg.evaluation.val_split)

    #     # get dataloaders
    #     train_dl, val_dl = get_dataloader(train_ds, val_ds, cfg.training.batch_size)









