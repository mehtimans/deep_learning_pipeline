"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: April 2026
"""

import pandas as pd 
import os
import json 
from typing import Tuple

import torch 
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

from deep_learning_course.models import MLPNetwork
from deep_learning_course.configs import HW2cfg
from deep_learning_course.utils import get_args, set_seed, update_cfg_from_args, class_to_dict
from deep_learning_course.utils import split_dataset, get_dataloader, compute_normalization_stats, get_log_dir, save_model_jit
from deep_learning_course.utils import get_loss, get_optimizer
from deep_learning_course.pipeline import Trainer
from deep_learning_course import DEEP_LEARNING_COURSE_RESOURCES_DIR, DEEP_LEARNING_COURSE_ROOT_DIR

def Dataset_generation(data_gen_cfg, val_split)-> Tuple[Dataset, Dataset]:

    means_ = torch.linspace(data_gen_cfg.mean_range[0], 
                           data_gen_cfg.mean_range[1], 
                           data_gen_cfg.num_classes)
    variance = torch.tensor(data_gen_cfg.variance, dtype=torch.float32)
    std_ = torch.sqrt(variance)

    train_ds_list = []
    val_ds_list = []

    for class_id, mu in enumerate(means_):

        samples_i = torch.normal(
            mean=mu,
            std=std_,
            size=(
                data_gen_cfg.samples_per_class,
                data_gen_cfg.image_dim,
                data_gen_cfg.image_dim
            )
        )
        labels_i = torch.full((data_gen_cfg.samples_per_class,), class_id)
        
        train_ds_i, val_ds_i = split_dataset(samples_i, labels_i, val_split)
        train_ds_list.append(train_ds_i)
        val_ds_list.append(val_ds_i)

    train_ds = ConcatDataset(train_ds_list)
    val_ds = ConcatDataset(val_ds_list)
    
    return train_ds, val_ds

if __name__ == "__main__":
    
    # get args
    args = get_args()
    
    # final configuration 
    cfg = update_cfg_from_args(args, HW2cfg())
    cfg_dict = class_to_dict(cfg)
    print(json.dumps(cfg_dict, indent=4))

    # set seed 
    cfg.seed = set_seed(cfg.seed)

    # get logging directory  
    cfg.logger.log_dir = get_log_dir(cfg.logger.log_dir, cfg.logger.train_label)

    # load and split dataset 
    train_ds, val_ds = Dataset_generation(cfg.datageneration, cfg.evaluation.val_split)
    print("Train dataset size:", len(train_ds))
    print("Val dataset size:", len(val_ds))    
    # x, y = train_ds[0] 
    # print(x.size(), y, type(y))  
    
    # get dataloaders
    train_dl, val_dl = get_dataloader(train_ds, val_ds, cfg.training.batch_size)
    # x, y = next(iter(train_dl))
    # print(x.size())

    # # compute_normalization_stats
    # mean, std = compute_normalization_stats(train_ds)
    # print("Feature mean:", mean.cpu().numpy())
    # print("Feature std: ", std.cpu().numpy())
    # print(f"output min: {Y.min().item():.4f}, max: {Y.max().item():.4f}, std: {Y.std().item():.4f}")

    # # load model
    # mlp_model = MLPNetwork(
    #                 num_inputs=X.size(1), 
    #                 num_outputs=Y.size(1),
    #                 network_hidden_dims = cfg.training.hidden_dims,
    #                 activation = cfg.training.activation)
    
    # mlp_model.set_normalization(mean, std)  

    # # loss function 
    # loss_fn = get_loss(cfg.training.loss, reduction="mean")

    # # Optimizer 
    # optimizer = get_optimizer(cfg.training.optimizer, mlp_model.parameters(), 
    #                           lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    
    # # Scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",          # because you monitor RMSE
    #     factor=0.5,          # LR *= factor when plateau
    #     patience=80,         # epochs with no improvement before reducing LR
    #     min_lr=1e-4,
    # )
    
    # # Train the model 
    # trainer = Trainer(
    #     mlp_model,
    #     train_dl,
    #     val_dl,
    #     optimizer,
    #     loss_fn,
    #     trainer_name=cfg.training.trainer.trainer_name,
    #     epochs=cfg.training.epochs,
    #     scheduler=scheduler, 
    #     device=cfg.device,
    #     noise_std=cfg.training.trainer.noise.noise_std,
    #     noise_frac=cfg.training.trainer.noise.noise_frac,
    #     metrics=cfg.training.trainer.metrics,
    #     monitor=cfg.training.trainer.monitor,
    #     mode=cfg.training.trainer.mode,
    #     early_stopping=cfg.training.trainer.early_stopping,
    #     patience=cfg.training.trainer.patience,
    #     log_dir=cfg.logger.log_dir 
    # )

    # best_model = trainer.train() # Note: best_model is stored on CPU for portability

    # # save model as a jit file
    # model_path = save_model_jit(best_model, cfg.logger.log_dir, cfg.logger.save_model_label)

    # # save config
    # config_path = os.path.join(cfg.logger.log_dir, "config.json")
    # with open(config_path, "w", encoding="utf-8") as f:
    #     json.dump(cfg_dict, f, indent=4)
    
    # print(f"config saved to {config_path}")



