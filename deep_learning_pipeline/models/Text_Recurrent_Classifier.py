"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: August 2026
"""

import torch
import torch.nn as nn

class TextRecurrentClassifier(nn.Module):
    """Text classifier with an embedding layer and recurrent backbone."""
    def __init__(self, 
                 backbone: nn.Module,
                 vocab_size: int,  
                 embedding_dim: int,
                 padding_idx: int = 0) -> None:

        super().__init__()

        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx)

        self.backbone = backbone

    def forward(self, x):

        # x: (B, T), containing token IDs

        lengths = (x != self.padding_idx).sum(dim=1)

        # (B, T) -> (B, T, E)
        x = self.embedding(x)

        return self.backbone(x, lengths)