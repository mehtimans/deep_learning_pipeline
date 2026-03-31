"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: March 2026
"""

import pandas as pd 
import os
import json 
from typing import Tuple

import torch 
from torchvision import datasets, transforms
from torch.utils.data import Dataset


from deep_learning_course.models import MLPNetwork, AutoEncoderNetwork 
from deep_learning_course.configs import HW1Q1cfg
from deep_learning_course.utils import get_args, set_seed, update_cfg_from_args, class_to_dict
from deep_learning_course.utils import get_dataloader, get_log_dir, save_model_jit
from deep_learning_course.utils import get_loss, get_optimizer
from deep_learning_course.pipeline import Trainer
from deep_learning_course import DEEP_LEARNING_COURSE_RESOURCES_DIR, DEEP_LEARNING_COURSE_ROOT_DIR

class SelfSupervisedDataset(Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __getitem__(self, idx):
        x, _ = self.base_ds[idx]
        return x, x

    def __len__(self):
        return len(self.base_ds)


def load_dataset(root_path: str)-> Tuple[Dataset, Dataset]:
    """
    Loads the MNIST dataset and applies transformations:
    - Converts PIL images to tensors (values in [0, 1])
    - Flattens each image into a 784-dimensional vector
    """

    # transform = transforms.PILToTensor() # without normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_ds = datasets.MNIST(
        root=root_path,
        train=True,
        download=False,
        transform=transform
    )

    val_ds = datasets.MNIST(
        root=root_path,
        train=False,
        download=False,
        transform=transform
    )
        
    return train_ds, val_ds

if __name__ == "__main__":

    # get args
    args = get_args()

    # final configuration 
    cfg = update_cfg_from_args(args, HW1Q1cfg())
    cfg_dict = class_to_dict(cfg)
    print(json.dumps(cfg_dict, indent=4))

    # set seed 
    cfg.seed = set_seed(cfg.seed)

    # get logging directory  
    cfg.logger.log_dir = get_log_dir(cfg.logger.log_dir, cfg.logger.train_label)

    # load dataset 
    data_root_path = os.path.join(DEEP_LEARNING_COURSE_RESOURCES_DIR, "data")
    train_ds, val_ds = load_dataset(data_root_path)
    print("Train size:", len(train_ds))
    print("Validation size:", len(val_ds))
    # labels = [y for _, y in val_ds]
    # x, y = train_ds[0] 
    # print(x.size(), type(y))     
    # print(labels)

    # get dataloaders
    train_dl, val_dl = get_dataloader(train_ds, val_ds, cfg.training.batch_size)
    # x, y = next(iter(train_dl))
    # print(x.size())

    ## Training Autoencoder Networks 
    Autoenc_train_ds = SelfSupervisedDataset(train_ds)
    Autoenc_val_ds = SelfSupervisedDataset(val_ds)
    x0, _ = Autoenc_train_ds[0]

    Autoenc_train_dl, Autoenc_val_dl = get_dataloader(Autoenc_train_ds, Autoenc_val_ds, cfg.training.batch_size)
    
    autoencoder_model = AutoEncoderNetwork(
        num_inputs=x0.numel(),
        latent_dim=cfg.training.latent_size,
        encoder_hidden_dims=cfg.training.encoder_hidden_dims,
        decoder_hidden_dims=cfg.training.decoder_hidden_dims,
        activation=cfg.training.activation
    ).to(cfg.device)

    # loss function 
    loss_fn = get_loss(cfg.training.loss, reduction="mean")

    # Optimizer 
    optimizer = get_optimizer(cfg.training.optimizer, autoencoder_model.parameters(), 
                              lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
 
    # Train the model 
    trainer = Trainer(
        autoencoder_model,
        Autoenc_train_dl,
        Autoenc_val_dl,
        optimizer,
        loss_fn,
        epochs=cfg.training.epochs,
        device=cfg.device,
        noise_std=cfg.training.noise.noise_std,
        noise_frac=cfg.training.noise.noise_frac,
        early_stopping=cfg.training.early_stopping,
        patience=cfg.training.patience,
        log_dir=cfg.logger.log_dir 
    )

    print("Starting autoencoder training...")
    best_autoencoder_model = trainer.train() # Note: best_autoencoder_model is stored on CPU for portability

    # save model as a jit file
    model_path = save_model_jit(best_autoencoder_model, cfg.logger.log_dir)

    # save config
    config_path = os.path.join(cfg.logger.log_dir, "autoencoder_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=4)
    
    print(f"config saved to {config_path}")