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
from typing import Type, List, Optional
from torch.nn.utils.rnn import pack_padded_sequence
from abc import ABC, abstractmethod

from deep_learning_pipeline.models import MLPNetwork

class BaseRecurrentNetwork(nn.Module, ABC):
    """Base class for recurrent sequence models."""
    def __init__(self,
                  recurrent_cls: Type[nn.Module],
                  num_inputs: int,
                  num_outputs: int,
                  hidden_size: int = 64,
                  num_layers: int = 4,
                  mlp_network_hidden_dims: Optional[List[int]] = None,
                  mlp_activation: str = "relu",
                  **kwargs):

        super().__init__()

        if kwargs:
            print(f"{self.__class__.__name__}.__init__ got unexpected "
                  f"arguments, which will be ignored: {list(kwargs.keys())}")
            
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # normalization buffers 
        self.register_buffer("x_mean", torch.zeros(num_inputs))
        self.register_buffer("x_std", torch.ones(num_inputs))
        self.normalize_inputs = False

        # Recurrent module
        self.recurrent = recurrent_cls(input_size=self.num_inputs, hidden_size=self.hidden_size, 
                                       num_layers=self.num_layers, batch_first=True)

        # Output head, Fully connected layers
        if mlp_network_hidden_dims is None:
            mlp_network_hidden_dims = []
        self.mlp_net = MLPNetwork(
            num_inputs=self.hidden_size, 
            num_outputs=num_outputs,
            network_hidden_dims = mlp_network_hidden_dims,
            activation = mlp_activation)

        # initialize weights
        self.apply(_orthogonal_init)

    @abstractmethod
    def extract_last_hidden(self, hidden):
        """Extract the final hidden state from the recurrent module output."""
        raise NotImplementedError

    @torch.no_grad()
    def set_normalization(self, mean, std, eps: float = 1e-6):
        """Store per-feature mean/std (from TRAIN split). Enables normalization in forward()."""
        mean_t = torch.as_tensor(mean, dtype=self.x_mean.dtype, device=self.x_mean.device)
        std_t  = torch.as_tensor(std,  dtype=self.x_std.dtype,  device=self.x_std.device).clamp_min(eps)
        self.x_mean.copy_(mean_t)
        self.x_std.copy_(std_t)
        self.normalize_inputs = True 

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """
        x: (batch_size, seq_len, num_inputs)
        returns: (batch_size, num_outputs)
        """
        # x: (batch_size, seq_len, num_inputs)
        if self.normalize_inputs:
            x = (x - self.x_mean) / self.x_std

        # Ignore padded time steps when sequence lengths are provided.
        if lengths is not None:
            x = pack_padded_sequence(
                x, 
                lengths.cpu(), 
                batch_first=True,
                enforce_sorted=False)
        
        # Recurrent forward pass
        _, hidden = self.recurrent(x)

        # Extract the last-layer hidden representation.
        last_hidden = self.extract_last_hidden(hidden)

        # Output head
        return self.mlp_net(last_hidden)  
        

def _orthogonal_init(m: nn.Module):
    """Apply orthogonal initialization to recurrent and linear layers."""
    if isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
