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
from typing import List, Optional

from .Base_Recurrent_Network import BaseRecurrentNetwork

class VanillaRNNNetwork(BaseRecurrentNetwork):
    def __init__ (self,
                num_inputs: int,
                num_outputs: int,
                hidden_size: int = 64,
                num_layers: int = 4,
                mlp_network_hidden_dims: Optional[List[int]] = None,
                mlp_activation: str = "relu",
                **kwargs):
        
        super().__init__(
            recurrent_cls = nn.RNN,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            mlp_network_hidden_dims=mlp_network_hidden_dims,
            mlp_activation=mlp_activation,
            **kwargs)

    def _extract_last_hidden(self, hidden):
        """Extract the final hidden state from the recurrent module output."""
        return hidden[-1]
