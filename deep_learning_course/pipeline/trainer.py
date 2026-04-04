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

from deep_learning_course.utils import METRIC_REGISTRY

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        optimizer,
        loss_fn,
        *,
        trainer_name: str = "run",
        epochs: int = 50,
        scheduler=None, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        noise_std: float = 0.0,
        noise_frac: float = 0.0,
        metrics: list = None, 
        monitor: str = "rmse",
        mode: str = "min",
        early_stopping: bool = True,
        patience: int = 100,
        log_dir: str  = "./logs",
        enable_plots:bool = True
    ):
        # Core training components
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        # Training configuration
        self.trainer_name = trainer_name
        self.epochs = epochs
        self.device = device

        # Optional noise injection for input regularization
        self.noise_std = noise_std
        self.noise_frac = noise_frac
        
        metrics = metrics or []
        self.metrics = []
        for m in metrics:
            if isinstance(m, str):
                if m not in METRIC_REGISTRY:
                    raise ValueError(f"unknown metric: {m}")
                self.metrics.append(METRIC_REGISTRY[m])
            else:
                raise ValueError(f"metric must be a string, got {type(m).__name__}")
        metric_names = [m.__name__ for m in self.metrics]
        
        # Metric used for model selection / early stopping
        self.monitor = monitor
        self.mode = mode.lower()
        if self.monitor not in metric_names:
            raise ValueError(f"monitor '{self.monitor}' not found in metrics {metric_names}")

        # Dataset sizes used for averaging losses
        self.n_train = len(train_dl.dataset)
        self.n_val = len(val_dl.dataset)


        # store the best model observed during training
        self.best_model = copy.deepcopy(self.model).cpu() 
        
        if self.mode == "min":
            self.best_score = float("inf")
        elif self.mode == "max":
            self.best_score = float("-inf")
        else: 
            raise ValueError (f"mode must be max or min, got {self.mode}")

        # early stopping parameters
        self.early_stopping = early_stopping
        self.patience = patience
        self.patience_counter = 0

        # Logging directory
        self.log_dir = log_dir

        # History container for metrics and losses
        self.history = {
            "epoch": [],
            "train_loss": [], "val_loss": []
        }
        for m in self.metrics:
            name = m.__name__
            self.history[f"train_{name}"] = []
            self.history[f"val_{name}"] = []
        
        # Track learning rate if scheduler is used
        if self.scheduler is not None:
            self.history["lr"] = []

        # Enable/disable automatic plotting
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
            
            # Run one training and validation pass
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
 
             # Extract monitored metric value
            monitor_value = val_metrics[self.monitor]
            monitor_value = monitor_value.item() if torch.is_tensor(monitor_value) else monitor_value

            # Set Scheduler
            lr_str = ""
            if self.scheduler is not None:
                
                # ReduceLROnPlateau requires the monitored metric
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()
                # Log learning rate after stepping
                lr = self.optimizer.param_groups[0]["lr"]
                self.history["lr"].append(lr)
                lr_str = f" | LR: {lr:.6f}"

            # Store history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            for name, value in train_metrics.items():
                self.history[f"train_{name}"].append(value.item()) 
            for name, value in val_metrics.items():
                self.history[f"val_{name}"].append(value.item())

            # Determine if monitored metric improved
            improved = (
                (self.mode == "min" and monitor_value < self.best_score) or
                (self.mode == "max" and monitor_value > self.best_score)
            )
            
            # dynamic metric printing
            train_str = " | ".join(
                f"{key}: {(value.item() if torch.is_tensor(value) else value):.4f}"
                for key, value in train_metrics.items()
            )
            
            val_str = " | ".join(
                f"{key}: {(value.item() if torch.is_tensor(value) else value):.4f}"
                for key, value in val_metrics.items()
            )
            
            # Best model tracking 
            if improved:
                self.best_score = monitor_value
                self.best_model = copy.deepcopy(self.model).to("cpu")
                self.patience_counter = 0
                print(f"Epoch {epoch:03d} | Train [{train_str}] | Val [{val_str}]{lr_str} | Best Model Saved")
            else:
                self.patience_counter += 1
                print(f"Epoch {epoch:03d} | Train [{train_str}] | Val [{val_str}]{lr_str}")
            
            
            if self.early_stopping and self.patience_counter >= self.patience:
                print("Early stopping triggered")
                break

        print(f"\nBest Validation {self.monitor}: {self.best_score:.4f}")
        
        # Save logs and generate plots
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
        preds_all = []
        targets_all = []


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

            # Forward pass
            pred = self.model(xb_noisy)
            loss = self.loss_fn(pred, yb)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Accumulate batch results
            total_loss += loss.item() * xb.size(0)
            preds_all.append(pred.detach())
            targets_all.append(yb.detach())

        train_loss = total_loss / self.n_train

        # full-epoch metric computation
        preds = torch.cat(preds_all)
        targets = torch.cat(targets_all)

        metric_results = {}
        for metric in self.metrics:
            metric_results[metric.__name__] = metric(preds, targets)

        return train_loss, metric_results


    @torch.no_grad()
    def validate_epoch(self):
        """
        Run one validation epoch without gradient updates.
        """
        
        # Switch to evaluation mode
        self.model.eval()
        
        total_loss = 0.0
        preds_all = []
        targets_all = []

        for xb, yb in self.val_dl:
            xb, yb = xb.to(self.device), yb.to(self.device)

            pred = self.model(xb)
            loss = self.loss_fn(pred, yb)
            
            total_loss += loss.item() * xb.size(0)
            preds_all.append(pred.detach())
            targets_all.append(yb.detach())

        val_loss = total_loss / self.n_val

        # full-epoch metric computation    
        preds = torch.cat(preds_all)
        targets = torch.cat(targets_all)

        metric_results = {}
        for metric in self.metrics:
            metric_results[metric.__name__] = metric(preds, targets)
        
        return val_loss, metric_results


    def save_log(self, dict_history: dict, log_dir=None):
        """
        Save the metrics history to a CSV file derived from self.history.
        """

        out_dir = log_dir if log_dir is not None else self.log_dir
        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(dict_history)

        out_path = os.path.join(out_dir, f"{self.trainer_name}_training_log.csv")
        tmp_path = out_path + ".tmp"

        try:
            df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, out_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save training log CSV to {out_path}") from e

        print(f"Training log saved to {out_path}")
        self.log_path = out_path

    def plotter(self, data, x_label="epoch", log_dir=None, show=False):
        """
        Generate and save plots for all tracked metrics.
        """

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
            # plt.title(f"{column} over {x_label}")

            plt.grid(True)
            plt.tight_layout()

            fig_path = os.path.join(out_dir, f"{self.trainer_name}_{column}.png")
            plt.savefig(fig_path)
            # print(f"Saved: {fig_path}")

            if show:
                plt.show(block=True)

            plt.close()
        
        print(f"Training plots saved to {out_dir}")

