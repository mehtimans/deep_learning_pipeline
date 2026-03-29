import numpy as np
import pandas as pd 
import argparse
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import torch
import json 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from actuator_network_MLP import ActuatorNetworkMLP

def train(
    model: nn.Module, 
    X: torch.Tensor, 
    Y: torch.Tensor, 
    *,
    val_split: float = 0.2,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    noise_std: float = 0.0, noise_frac: float = 0.0):    

    # load model and data
    model = model.to(device)
    X, Y = X.to(device), Y.to(device)
    
    # split
    ds = TensorDataset(X, Y)
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # normalize inputs 
    train_indices = train_ds.indices
    train_X = X[train_indices]                 # (N_train, D)
    mean = train_X.mean(dim=0)
    std  = train_X.std(dim=0)
    model.set_normalization(mean, std)  
    print("Feature mean:", mean.cpu().numpy())
    print("Feature std: ", std.cpu().numpy())
    print(f"output min: {Y.min().item():.4f}, max: {Y.max().item():.4f}, std: {Y.std().item():.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    # Track best model
    best_val_rmse = float("inf")
    best_model = copy.deepcopy(model).cpu()

    for epoch in range(1, epochs + 1):
        # --------- train ----------
        model.train()
        train_sum = 0.0
        for xb, yb in train_dl:
            # ----- add noise to inputs -----
            xb_noisy = xb
            if noise_std > 0.0 or noise_frac > 0.0:
                # build per-feature std in RAW units
                if noise_frac > 0.0 and hasattr(model, "x_std"):
                    # use stored TRAIN-set std if available
                    feat_std = model.x_std.to(xb.device)  # shape (D,)
                    per_feat = noise_frac * feat_std
                else:
                    per_feat = torch.full_like(xb, fill_value=noise_std)  # same std for all features
                # sample Gaussian noise per element
                noise = torch.randn_like(xb) * per_feat
                xb_noisy = xb + noise
            # -------------------------------
            pred = model(xb_noisy)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * xb.size(0)

        train_mse = train_sum / len(train_ds)
        train_rmse = train_mse ** 0.5
        
        # --------- validate ----------
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb.to(device))
                val_sum += loss_fn(pred, yb.to(device)).item() * xb.size(0)
        val_mse = val_sum / len(val_ds)
        val_rmse = val_mse ** 0.5

        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = copy.deepcopy(model).cpu()
            print(f"epoch {epoch:03d} | train RMSE: {train_rmse:.6f} | validate RMSE {val_rmse:.6f} | best model saved")
        else :
            print(f"epoch {epoch:03d} | train RMSE: {train_rmse:.6f} | validate RMSE {val_rmse:.6f}")


    
    print(f"Best validation RMSE: {best_val_rmse:.6f}")
    return best_model

def get_args():
    parser = argparse.ArgumentParser()
    # NOTICE: All 'default=...' arguments are removed. They will default to None.
    parser.add_argument("--val_split", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--noise_std", type=float, help="Absolute input noise (units of X)")
    parser.add_argument("--noise_frac", type=float, help="Fraction of per-feature std")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    return args
 
def save_model(model):
    model = copy.deepcopy(model).to("cpu").eval()
    
    # directory to save
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
    log_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "jit_models")
    log_dir = os.path.join(log_root, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, 'JIT_model.pt')

    # save as jit
    scripted_module = torch.jit.script(model)
    scripted_module.save(model_path)
    print(f"Model saved to {model_path}")

    return log_dir

if __name__ == "__main__":
    
    # get args
    args = get_args()

    # seting seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load data
    base_dir = os.path.dirname(os.path.realpath(__file__))
    csv_folder = os.path.join(base_dir, "csv_files", "datasets")
    csv_path = os.path.join(csv_folder, "merged_data.csv")
    df = pd.read_csv(csv_path, sep=",")

    X = torch.tensor(df[["e_t", "v_t", "e_t_10", "v_t_10", "e_t_20", "v_t_20"]].values, dtype=torch.float32)
    Y = torch.tensor(df[["y"]].values, dtype=torch.float32)
    
    print("X shape", X.size())
    print("Y shape", Y.size())
    
    hidden_dims = [32, 32, 32]
    activation_function = "softsign"
    # load model
    mlp_model = ActuatorNetworkMLP(
                    num_inputs=X.size(1), 
                    num_outputs=Y.size(1),
                    network_hidden_dims = hidden_dims,
                    activation = activation_function)
    
    # train model
    best_model = train(
        mlp_model, X, Y,
        val_split = args.val_split,
        epochs = args.epochs,
        batch_size = args.batch_size,
        lr = 3e-4,
        noise_std = args.noise_std,
        noise_frac = args.noise_frac,
    )
    
    # save model
    log_dir = save_model(best_model)

    metadata = {
        "taps_steps": [0, 10, 20],                     # Step offsets used for stacking
        "dt_sec": 0.0011,                              # Sample time in seconds
        "features": ["e_t", "v_t", "e_t_10", "v_t_10", "e_t_20", "v_t_20"],
        "label": "joint_torque",                      # What Y represents
        "normalization_mean": best_model.x_mean.cpu().tolist(),  # From train split
        "normalization_std": best_model.x_std.cpu().tolist(),
        "activation": activation_function,                     # Network activation used
        "hidden_layers": hidden_dims,                # MLP architecture
        "output_dim": Y.size(1),
        "input_dim": X.size(1),
        "loss": "SmoothL1Loss(beta=1.0)",             # Loss used in training
        "optimizer": "Adam(lr=3e-4)",                 # Optimizer config
        "noise_injected": {
            "noise_std": args.noise_std,
            "noise_frac": args.noise_frac
        },
        "val_split": args.val_split,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    with open(os.path.join(log_dir, "info.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Visualize prediction error histogram
    best_model.eval()
    with torch.no_grad():
        preds = best_model(X.to(best_model.x_mean.device)).cpu().squeeze()
        targets = Y.cpu().squeeze()
        errors = (preds - targets).numpy()

    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=100, color='skyblue', edgecolor='black')
    plt.title("prediction Error Histogram")
    plt.xlabel("Error (Nm)")
    plt.ylabel("count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "error_histogram.png"))
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(targets.numpy(), label="Actual Torque", linewidth=1)
    plt.plot(preds.numpy(), label="Predicted Torque", linewidth=1, alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Torque (Nm)")
    plt.title("Predicted vs Actual Torque Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "pred_vs_actual_timeseries.png"))
    plt.show()
