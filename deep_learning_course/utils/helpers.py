"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: March 2026
"""

import os
import torch
import numpy as np
import random
import argparse

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed: int):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    # print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed

def get_args():
    parser = argparse.ArgumentParser()
    # NOTICE: All 'default=...' arguments are removed. They will default to None.
    parser.add_argument("--val_split", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--hidden_dims", type=int, nargs="+",
                    help="Hidden layer sizes, e.g. --hidden_dims 64 64")
    parser.add_argument("--add_noise", type=lambda x: x.lower() == "true",
                    help="Enable or disable input noise (true/false)")
    parser.add_argument("--noise_std", type=float, help="Absolute input noise (units of X)")
    parser.add_argument("--noise_frac", type=float, help="Fraction of per-feature std")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    return args
 
def update_cfg_from_args(args, cfg):

    if args.seed is not None:
        cfg.seed = args.seed
    
    if args.device is not None:
        cfg.device = args.device

    # training
    if args.epochs is not None:
        cfg.training.epochs = args.epochs

    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size

    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate

    if args.hidden_dims is not None:
        cfg.training.hidden_dims = args.hidden_dims

    if args.add_noise is not None:
        cfg.training.add_noise = args.add_noise

    # noise config
    if args.noise_std is not None:
        cfg.training.noise.noise_std = args.noise_std

    if args.noise_frac is not None:
        cfg.training.noise.noise_frac = args.noise_frac

    # evaluation
    if args.val_split is not None:
        cfg.evaluation.val_split = args.val_split
    
    # enforce rule
    if not cfg.training.add_noise:
        cfg.training.noise.noise_std = 0.0
        cfg.training.noise.noise_frac = 0.0

    return cfg
