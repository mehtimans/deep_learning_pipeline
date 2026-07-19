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
from deep_learning_course.configs import AutoEncoderCfg
from deep_learning_course.utils import get_args, set_seed, update_cfg_from_args, class_to_dict
from deep_learning_course.utils import get_dataloader, get_log_dir, save_model_jit, count_trainable_params
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

class LatentDataset(Dataset):
    def __init__(self, z, y):
        self.z = z
        self.y = y

    def __getitem__(self, idx):
        return self.z[idx], self.y[idx]

    def __len__(self):
        return len(self.z)
    
def build_latent_dataset(model, dataloader, device):
    model.eval()
    zs = []
    ys = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            z = model.encode(xb)
            zs.append(z.cpu())
            ys.append(yb.cpu())

    zs = torch.cat(zs, dim=0)
    ys = torch.cat(ys, dim=0)
    return LatentDataset(zs, ys)

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
    cfg = update_cfg_from_args(args, AutoEncoderCfg())
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

    if cfg.evaluation.load_autoencoder == -1:
        ## Training Autoencoder Networks 
        Autoenc_train_ds = SelfSupervisedDataset(train_ds)
        Autoenc_val_ds = SelfSupervisedDataset(val_ds)
        x0, _ = Autoenc_train_ds[0]

        Autoenc_train_dl, Autoenc_val_dl = get_dataloader(Autoenc_train_ds, Autoenc_val_ds, cfg.training.autoencoder_batch_size)
        
        autoencoder_model = AutoEncoderNetwork(
            num_inputs=x0.numel(),
            latent_dim=cfg.training.latent_size,
            encoder_hidden_dims=cfg.training.encoder_hidden_dims,
            decoder_hidden_dims=cfg.training.decoder_hidden_dims,
            activation=cfg.training.activation
        ).to(cfg.device)

        autoencoder_num_params = count_trainable_params(autoencoder_model)
        print(f"Autoencoder has {autoencoder_num_params:,} trainable parameters")
        
        # loss function 
        autoencoder_loss_fn = get_loss(cfg.training.autoencoder_loss, reduction="mean")

        # Optimizer 
        autoencoder_optimizer = get_optimizer(cfg.training.optimizer, autoencoder_model.parameters(), 
                                lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        
        # Scheduler
        autoencoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            autoencoder_optimizer,
            mode="min",          # because you monitor RMSE
            factor=0.5,          # LR *= factor when plateau
            patience=10,         # epochs with no improvement before reducing LR
            min_lr=1e-4,
        )
    
        # Train the model 
        autoencoder_trainer = Trainer(
            autoencoder_model,
            Autoenc_train_dl,
            Autoenc_val_dl,
            autoencoder_optimizer,
            autoencoder_loss_fn,
            trainer_name=cfg.training.trainer.autoencoder_trainer_name,
            epochs=cfg.training.epochs,
            # scheduler=autoencoder_scheduler, 
            device=cfg.device,
            noise_std=cfg.training.trainer.noise.autoencoder_noise_std,
            noise_frac=cfg.training.trainer.noise.autoencoder_noise_frac,
            metrics=cfg.training.trainer.autoencoder_metrics,
            monitor=cfg.training.trainer.autoencoder_monitor,
            mode=cfg.training.trainer.autoencoder_mode,
            early_stopping=cfg.training.trainer.early_stopping,
            patience=cfg.training.trainer.autoencoder_patience,
            log_dir=cfg.logger.log_dir 
        )

        print("Starting autoencoder training...")
        best_autoencoder_model = autoencoder_trainer.train() # Note: best_autoencoder_model is stored on CPU for portability

        # save model as a jit file
        autoencoder_model_path = save_model_jit(best_autoencoder_model, cfg.logger.log_dir, cfg.logger.autoencoder_save_model_label)


    else:
        try: 
            best_autoencoder_model = torch.jit.load(cfg.evaluation.load_autoencoder)
        
        except Exception as e:
            raise ValueError(f"Invalid autoencoder path: {cfg.evaluation.load_autoencoder}") from e


    best_autoencoder_model = best_autoencoder_model.to(cfg.device)
    
    # freeze autoencoder parameters
    for param in best_autoencoder_model.parameters():
        param.requires_grad = False


    ## Training Classifier network
    # build Classifier Dataset 
    train_class_ds = build_latent_dataset(best_autoencoder_model, train_dl, cfg.device)
    val_class_ds = build_latent_dataset(best_autoencoder_model, val_dl, cfg.device)
    x1, _ = train_class_ds[0]

    train_class_dl, val_class_dl = get_dataloader(train_class_ds, val_class_ds, cfg.training.classifier_batch_size)
    
    # load model
    classifier_mlp_model = MLPNetwork(
                    num_inputs=x1.numel(), 
                    num_outputs=cfg.training.classifier_num_outputs,
                    network_hidden_dims = cfg.training.classifier_hidden_dims,
                    activation = cfg.training.activation)
    
    classifier_num_params = count_trainable_params(classifier_mlp_model)
    print(f"Classifier has {classifier_num_params:,} trainable parameters")
    
    # loss function 
    classifier_loss_fn = get_loss(cfg.training.classifier_loss, reduction="mean")

    # Optimizer 
    classifier_optimizer = get_optimizer(cfg.training.optimizer, classifier_mlp_model.parameters(), 
                            lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    # Train the model 
    classifier_trainer = Trainer(
        classifier_mlp_model,
        train_class_dl,
        val_class_dl,
        classifier_optimizer,
        classifier_loss_fn,
        trainer_name=cfg.training.trainer.classifier_trainer_name,
        epochs=cfg.training.epochs,
        device=cfg.device,
        noise_std=cfg.training.trainer.noise.classifier_noise_std,
        noise_frac=cfg.training.trainer.noise.classifier_noise_frac,
        metrics=cfg.training.trainer.classifier_metrics,
        monitor=cfg.training.trainer.classifier_monitor,
        mode=cfg.training.trainer.classifier_mode,
        early_stopping=cfg.training.trainer.early_stopping,
        patience=cfg.training.trainer.classifier_patience,
        log_dir=cfg.logger.log_dir 
    )

    print("Starting classifier training...")
    best_classifier_model = classifier_trainer.train() # Note: best_classifier_model is stored on CPU for portability

    # save model as a jit file
    classifier_model_path = save_model_jit(best_classifier_model, cfg.logger.log_dir, cfg.logger.classifier_save_model_label)

    # save config
    cfg_dict["parameter_counts"] = {
        "autoencoder": autoencoder_num_params,
        "classifier": classifier_num_params
    }
    config_path = os.path.join(cfg.logger.log_dir, "autoencoder_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=4)
    
    print(f"config saved to {config_path}")
