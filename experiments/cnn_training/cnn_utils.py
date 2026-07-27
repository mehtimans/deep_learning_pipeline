"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: April 2026
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

import torch 
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D

from deep_learning_pipeline.utils import split_dataset, pca_torch

def Dataset_generation(data_gen_cfg, val_split)-> Tuple[Dataset, Dataset]:
    """Generate synthetic image dataset with class labels and train/validation split."""

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
                data_gen_cfg.num_channels,
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

def compute_normalization_stats(train_ds: Dataset) -> Tuple[Tensor, Tensor]:
    """Compute channel-wise mean and standard deviation from training data."""
   
    X_list = []

    for subset in train_ds.datasets:
        X_full = subset.dataset.tensors[0] # TensorDataset data
        idx = subset.indices # indices of this subset
        X_list.append(X_full[idx]) # select subset samples

    X = torch.cat(X_list, dim=0)

    mean = X.mean(dim=(0, 2, 3))
    std = X.std(dim=(0, 2, 3)).clamp(min=1e-8)

    return mean, std

def compute_covariance_matrix(train_ds: Dataset):
    """Compute and visualize covariance between class mean representations."""
    
    X_list = []
    y_list = []

    for subset in train_ds.datasets:
        X_full, y_full = subset.dataset.tensors
        idx = subset.indices
        X_list.append(X_full[idx])
        y_list.append(y_full[idx])

    X = torch.cat(X_list, dim=0)  # All images
    y = torch.cat(y_list, dim=0)  # All labels

    class_means = []
    num_classes = 10 

    for i in range(num_classes):

        class_samples = X[y == i] 
        class_samples_flat = class_samples.view(class_samples.shape[0], -1)
        mean_vec = torch.mean(class_samples_flat, dim=0)
        class_means.append(mean_vec.cpu().numpy())

    class_means_matrix = np.array(class_means)
    class_cov = np.cov(class_means_matrix)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(class_cov, cmap='Greens', interpolation='nearest')
    plt.colorbar(im)
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{class_cov[i, j]:.5f}", 
                    ha="center", va="center", color="black", fontsize=8)
    plt.title('Covariance Matrix Between Classes')
    plt.xlabel('Class ID')
    plt.ylabel('Class ID')
    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))

    plt.show()

def plot_mean_histograms(train_ds: Dataset):
    """Plot pixel-value distributions for representative samples of each class."""

    X_list = []
    y_list = []

    for subset in train_ds.datasets:
        X_full, y_full = subset.dataset.tensors
        idx = subset.indices
        X_list.append(X_full[idx])
        y_list.append(y_full[idx])

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    num_classes = 10
    plt.figure(figsize=(15, 8))

    for c in range(num_classes):

        # all samples of class c
        class_samples = X[y == c]
        # take ONE sample from class c
        sample = class_samples[0]
        sample_flat = sample.view(-1).cpu().numpy()

        # plot its histogram
        plt.subplot(2, 5, c + 1)
        plt.hist(sample_flat, bins=30, color='green', edgecolor='black')
        plt.title(f"Class {c}")
        plt.xlabel("Pixel value")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def mean_classifier_confusion_matrix(train_ds, test_ds):
    """Evaluate a nearest class-mean classifier and visualize its confusion matrix."""

    X_train_list, y_train_list = [], []
    for subset in train_ds.datasets:
        X_full, y_full = subset.dataset.tensors
        idx = subset.indices
        X_train_list.append(X_full[idx])
        y_train_list.append(y_full[idx])
    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)

    num_classes = 10
    class_means = []
    for c in range(num_classes):
        class_samples = X_train[y_train == c]
        class_mean = class_samples.view(class_samples.shape[0], -1).mean(dim=0)
        class_means.append(class_mean)
    class_means = torch.stack(class_means)  # shape [num_classes, features]

    X_test_list, y_test_list = [], []
    for subset in test_ds.datasets:
        X_full, y_full = subset.dataset.tensors
        idx = subset.indices
        X_test_list.append(X_full[idx])
        y_test_list.append(y_full[idx])
    X_test = torch.cat(X_test_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)

    X_test_flat = X_test.view(X_test.shape[0], -1)
    preds = torch.zeros_like(y_test)

    for i, sample in enumerate(X_test_flat):
        distances = torch.norm(class_means - sample, dim=1)  # Euclidean distance
        preds[i] = torch.argmin(distances)

    onfusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)
    for t, p in zip(y_test, preds):
        onfusion_matrix[t, p] += 1

    accuracy = (onfusion_matrix.diag().sum().float() / onfusion_matrix.sum().float()).item() * 100
    print(f"Mean Classifier Accuracy: {accuracy:.2f}%")

    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(onfusion_matrix, cmap='Greens')
    plt.title("Confusion Matrix with Mean Classifier")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.colorbar(label="Count")

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{onfusion_matrix[i, j].item()}", ha='center', va='center', color='black', fontsize=8)

    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device= "cuda", num_classes=10):
    """Evaluate model classification performance and generate confusion matrix."""

    # Initialize confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    # plot confusion matrix
    plt.figure(figsize=(8,6))
    plt.imshow(confusion_matrix.cpu(), cmap='Greens')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.colorbar(label="Count")

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{confusion_matrix[i, j].item()}",
                     ha='center', va='center', color='black', fontsize=8)

    plt.tight_layout()
    plt.show()

    # compute accuracy
    accuracy = torch.trace(confusion_matrix) / confusion_matrix.sum()
    print(f"Accuracy: {accuracy*100:.2f}%")

def plot_PCA(model, test_loader,  device= "cuda"):
    """Visualize model output representations using 2D PCA projection."""

    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():  
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)  
            if isinstance(outputs, tuple):
                output_t = outputs[0]
            else:
                output_t = outputs

            logits_list.append(output_t.cpu())
            labels_list.append(targets.cpu())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    features_2d = pca_torch(logits, n_components=2)
    features_2d_np = features_2d.numpy()
    labels_np = labels.numpy()

    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels_np)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for i, class_id in enumerate(unique_labels):
        idx = labels_np == class_id
        plt.scatter(
            features_2d_np[idx, 0],
            features_2d_np[idx, 1],
            color=colors(i),
            label=f"Class {class_id}",
            s=25,
            alpha=0.7
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Logit Features")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_PCA_3D(model, test_loader, device="cuda"):
    """Visualize model output representations using 3D PCA projection."""

    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                output_t = outputs[0]
            else:
                output_t = outputs

            logits_list.append(output_t.cpu())
            labels_list.append(targets.cpu())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    features_3d = pca_torch(logits, n_components=3)
    features_3d_np = features_3d.numpy()
    labels_np = labels.numpy()

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels_np)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for i, class_id in enumerate(unique_labels):
        idx = labels_np == class_id

        ax.scatter(
            features_3d_np[idx,0],
            features_3d_np[idx,1],
            features_3d_np[idx,2],
            color=colors(i),
            label=f"Class {class_id}",
            s=25,
            alpha=0.7
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_title("3D PCA of Logit Features ")

    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()


def cosine_test():
    """Verify cosine similarity calculation between normalized features and weights."""

    # Create dummy data
    B, D, C = 2, 128, 10
    features = torch.randn(B, D)
    weights = torch.randn(C, D)

    # Normalize 
    f = F.normalize(features, dim=1)
    W = F.normalize(weights, dim=1)

    cos_theta = torch.matmul(f, W.t())

    # Verification
    f0 = f[0]
    w0 = W[0]
    manual_cos = torch.dot(f0, w0)

    print(f"Matrix dot product result: {cos_theta[0, 0].item():.6f}")
    print(f"Manual dot product of normalized vectors: {manual_cos.item():.6f}")
    print(f"Are they equal? {torch.isclose(cos_theta[0, 0], manual_cos).item()}")

def evaluate_embedding_quality(model, dataloader, device="cuda", num_classes=None):
    """Analyze model predictions and learned feature representations by computing
       macro precision, macro recall, intra-class compactness, and inter-class separation."""
    
    model.eval()
    all_features, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)

            if isinstance(out, tuple):

                if len(out) == 3:
                    logits, features, _ = out
                else:
                    logits, features = out
            else:
                logits = out
                features = out  

            preds = logits.argmax(dim=1)
            all_features.append(features.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    features = torch.cat(all_features).numpy()
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    if num_classes is None:
        num_classes = int(labels.max()) + 1

    # Macro Precision & Macro Recall
    eps = 1e-12
    precisions = []
    recalls = []

    for c in range(num_classes):
        tp = np.logical_and(preds == c, labels == c).sum()
        fp = np.logical_and(preds == c, labels != c).sum()
        fn = np.logical_and(preds != c, labels == c).sum()

        precision_c = tp / (tp + fp + eps)
        recall_c    = tp / (tp + fn + eps)

        # If class does not appear in labels at all, skip from macro
        if (labels == c).sum() == 0:
            continue

        precisions.append(precision_c)
        recalls.append(recall_c)

    macro_precision = float(np.mean(precisions)) if precisions else np.nan
    macro_recall    = float(np.mean(recalls)) if recalls else np.nan

    # Intra-class distance
    intra = []
    for c in np.unique(labels):
        feats = features[labels == c]
        if len(feats) > 1:
            intra.extend(pdist(feats, metric='euclidean'))
    mean_intra = float(np.mean(intra)) if intra else np.nan

    # Inter-class distance
    centers = []
    for c in np.unique(labels):
        centers.append(features[labels == c].mean(axis=0))
    centers = np.stack(centers)
    mean_inter = float(np.mean(pdist(centers, metric='euclidean')))

    print("Macro Precision:", macro_precision)
    print("Macro Recall:", macro_recall)
    print("Mean inter-class distance:", mean_inter)
    print("Mean intra-class distance:", mean_intra)

def plot_class_center_angles(model, val_dl, device="cuda", title="Class Center Angles"):
    """Compute angular relationships between class embedding centers and visualize them."""

    model.eval()
    model.to(device)

    all_embs = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)
            y = y.to(device)

            _, emb, _ = model(x)   # change if your model returns (logits, emb)
            
            all_embs.append(emb.cpu())
            all_labels.append(y.cpu())

    embeddings = torch.cat(all_embs).numpy()
    labels = torch.cat(all_labels).numpy()

    classes = np.unique(labels)

    # compute class centers
    centers = []
    for c in classes:
        centers.append(embeddings[labels == c].mean(axis=0))
    centers = np.array(centers)

    # normalize centers
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    # cosine similarity using dot product
    cos_sim = centers @ centers.T
    cos_sim = np.clip(cos_sim, -1, 1)

    # convert to angles
    angles = np.degrees(np.arccos(cos_sim))

    # plot
    plt.figure(figsize=(6,5))
    plt.imshow(angles, cmap="viridis")
    plt.colorbar(label="Angle (degrees)")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.xticks(range(len(classes)))
    plt.yticks(range(len(classes)))
    plt.show()
