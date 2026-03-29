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
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        optimizer,
        loss_fn,
        *,
        epochs: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        noise_std: float = 0.0,
        noise_frac: float = 0.0,
        early_stopping: bool = True,
        patience: int = 100,
        log_dir: str  = "./logs",
        enable_plots:bool = True
    ):
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.epochs = epochs
        self.device = device

        self.noise_std = noise_std
        self.noise_frac = noise_frac

        self.n_train = len(train_dl.dataset)
        self.n_val = len(val_dl.dataset)

        # store the best model observed during training
        self.best_model = copy.deepcopy(self.model).cpu() 
        self.best_val_rmse = float("inf")
        
        # early stopping parameters
        self.early_stopping = early_stopping
        self.patience = patience
        self.patience_counter = 0

        self.log_dir = log_dir

        # History tracking
        self.history = {
            "epoch": [],
            "train_loss": [], "val_loss": [],
            "train_rmse": [], "val_rmse": [],
            "train_mse": [], "val_mse": [],
            "train_r2":   [], "val_r2":   []
        }

        self.enable_plots = enable_plots

    
    def train(self):
        """
        Main training loop.
        Runs training and validation for each epoch, logs metrics,
        tracks the best model based on validation RMSE, and
        applies early stopping if configured.
        """

        print(f"Starting training on {self.device}...")
        for epoch in range (1, self.epochs + 1):
            train_loss, train_mse, train_rmse, train_r2 = self.train_epoch()
            val_loss, val_mse, val_rmse, val_r2= self.validate_epoch()

            # add logs
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss), self.history["val_loss"].append(val_loss)
            self.history["train_rmse"].append(train_rmse), self.history["val_rmse"].append(val_rmse)
            self.history["train_mse"].append(train_mse), self.history["val_mse"].append(val_mse)
            self.history["train_r2"].append(train_r2), self.history["val_r2"].append(val_r2)

            # save best model
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse 
                self.best_model = copy.deepcopy(self.model).to("cpu")
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | "
                    f"Train R2: {train_r2:.4f} | Val R2: {val_r2:.4f} | Best Model Saved "
                )
                self.patience_counter = 0
            else:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | "
                    f"Train R2: {train_r2:.4f} | Val R2: {val_r2:.4f}"
                )
                self.patience_counter += 1
            
            if self.early_stopping and self.patience_counter >= self.patience:
                print("Early stopping triggered")
                break

        print(f"\nBest Validation RMSE: {self.best_val_rmse:.4f}")
        self.save_log(self.history)
        self.plotter(self.history)
        
        return self.best_model


    def train_epoch(self):
        """
        Perform one full training epoch.
        Updates model weights using backpropagation and
        computes training loss, MSE, RMSE, and R².
        """
        self.model.train()
        total_loss = 0.0
        ss_res = 0.0
        sum_y = 0.0
        sum_y2 = 0.0

        for xb, yb in self.train_dl:
            xb, yb = xb.to(self.device), yb.to(self.device)

            # add noise to inputs 
            xb_noisy = xb
            if self.noise_std > 0.0 or self.noise_frac > 0.0:
                # build per-feature std in RAW units
                if self.noise_frac > 0.0 and hasattr(self.model, "x_std"):
                    # use stored TRAIN-set std if available
                    feat_std = self.model.x_std.to(xb.device)  # shape (D,)
                    per_feat = self.noise_frac * feat_std
                else:
                    per_feat = self.noise_std  # same std for all features
                # sample Gaussian noise per element
                noise = torch.randn_like(xb) * per_feat
                xb_noisy = xb + noise

            self.optimizer.zero_grad()

            pred = self.model(xb_noisy)
            loss = self.loss_fn(pred, yb)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * xb.size(0)
            ss_res += torch.sum((yb - pred) ** 2).item()
            sum_y += torch.sum(yb).item()
            sum_y2 += torch.sum(yb ** 2).item()

        train_loss = total_loss / self.n_train

        train_mse = ss_res / self.n_train
        train_rmse = train_mse ** 0.5

        ss_tot = sum_y2 - (sum_y ** 2) / self.n_train
        train_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return train_loss, train_mse, train_rmse, train_r2


    def validate_epoch(self):
        """
        Run one validation epoch without gradient updates.
        """
        self.model.eval()
        total_loss = 0.0
        ss_res = 0.0
        sum_y = 0.0
        sum_y2 = 0.0

        with torch.no_grad():
            for xb, yb in self.val_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)

                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                
                total_loss += loss.item() * xb.size(0)
                ss_res += torch.sum((yb - pred) ** 2).item()
                sum_y += torch.sum(yb).item()
                sum_y2 += torch.sum(yb ** 2).item()

        val_loss = total_loss / self.n_val
        
        val_mse = ss_res / self.n_val
        val_rmse = val_mse ** 0.5

        ss_tot = sum_y2 - (sum_y ** 2) / self.n_val
        val_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return val_loss, val_mse, val_rmse, val_r2

    def save_log(self, dict_history: dict, log_dir=None):
        """
        Save the metrics history to a CSV file derived from self.history.
        """

        out_dir = log_dir if log_dir is not None else self.log_dir
        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(dict_history)

        out_path = os.path.join(out_dir, "training_log.csv")
        tmp_path = out_path + ".tmp"

        try:
            df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, out_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save training log CSV to {out_path}") from e

        print(f"Training log saved to {out_path}")
        self.log_path = out_path

    def plotter(self, data, x_label="epoch", log_dir=None, show=False):

        if not self.enable_plots:
            return
        
        out_dir = log_dir if log_dir is not None else self.log_dir
        os.makedirs(out_dir, exist_ok=True)
        
        columns = [k for k in data.keys() if k != x_label]

        for column in columns:
            plt.figure(figsize=(8, 4))
            plt.plot(data[x_label], data[column])

            plt.xlabel(x_label)
            plt.ylabel(column)
            # plt.title(f"{column} over f{x_label}")

            plt.grid(True)
            plt.tight_layout()

            fig_path = os.path.join(out_dir, f"{column}.png")
            plt.savefig(fig_path)
            # print(f"Saved: {fig_path}")

            if show:
                plt.show(block=True)

            plt.close()
        
        print(f"Training plots saved to {out_dir}")

