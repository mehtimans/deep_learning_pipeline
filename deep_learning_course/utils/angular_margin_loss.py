import torch
import torch.nn as nn
import torch.nn.functional as F

class AngularMarginLoss(nn.Module):
    def __init__(self, s=30.0, m=0.3, reduction="mean"):
        super().__init__()

        self.s = s
        self.m = m
        self.reduction = reduction

    def forward(self, features, weights, targets):
        """
        logits:   (B, C)
        features: (B, D)
        weights:  (C, D)
        targets:  (B,)
        """
        
        # Normalize features and weights
        f = F.normalize(features, dim=1) 
        W = F.normalize(weights, dim=1)

        # Cosine 
        cos_theta = torch.matmul(f, W.t()) # (B, C)

        # Mask for correct classes
        batch_size = features.size(0)
        mask = torch.zeros_like(cos_theta)
        mask.scatter_(1, targets.view(-1, 1).long(), 1.0)

        logits = self.s * (cos_theta - mask * self.m)

        loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        
        return loss

