"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: April 2026
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F


class IAMLoss(nn.Module):
    def __init__(self, beta=0.5, s=30.0, reduction="mean"):
        super().__init__()

        self.beta = beta
        self.s = s
        self.reduction = reduction 
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits, features, weights, targets):
        """
        logits:   (B, C)
        features: (B, D)
        weights:  (C, D)
        targets:  (B,)
        """

        # Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(logits, targets)

        # normalize features and weights 
        f = F.normalize(features, dim=1)
        W = F.normalize(weights, dim=1)

        # Cosine
        cos_theta = torch.matmul(f, W.t())

        # scaled angle 
        scaled = self.s * cos_theta
        exp_scaled = torch.exp(scaled) # (B, C)

        # denominator 
        denom = exp_scaled.sum(dim=1) # (B,)

        # mask for correct class
        mask = F.one_hot(targets, num_classes=W.size(0)).bool() # (B, C)

        # remove correct class
        wrong_scaled = exp_scaled.masked_fill(mask, 0.0) # (B, C)

        # numerator
        numer = wrong_scaled.sum(dim=1)/(W.size(0) - 1)

        iam = torch.log(numer/denom) # (B,)

        if self.reduction == "mean":
            iam = iam.mean()
        elif self.reduction == "sum":
            iam = iam.sum()

        loss = ce_loss + self.beta * iam

        return loss
    

