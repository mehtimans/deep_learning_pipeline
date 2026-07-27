"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: March 2026
"""


import torch
import torch.nn as nn
from typing import List

from deep_learning_pipeline.utils import get_activation

class LSTMNetwork(nn.Module):
    def __init__ (self,
                  num_inputs: int,
                  num_outputs: int,
                  hidden_size: int = 64,
                  num_layers: int = 4,
                  activation: str = 'softsign',
                  **kwargs):
        
        if kwargs:
            print("LSTMNetwork.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(LSTMNetwork, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # normalization buffers 
        self.register_buffer("x_mean", torch.zeros(num_inputs))
        self.register_buffer("x_std", torch.zeros(num_inputs))
        self.normalize_inputs = False

        # activation function 
        self.activation = get_activation(activation)

        # LSTM module
        self.lstm = nn.LSTM(input_size=self.num_inputs, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        # MLP head
        self.linear = nn.Linear(self.hidden_size, self.num_outputs)

        self.apply(_orthogonal_init)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)

        print(self)
        # print(f"Actuator Network LSTM: LSTM({num_inputs}→{hidden_size}×{num_layers}), Linear({hidden_size}→{num_outputs})")

    @torch.no_grad()
    def set_normalization(self, mean, std, eps: float = 1e-6):
        """Store per-feature mean/std (from TRAIN split). Enables normalization in forward()."""
        mean_t = torch.as_tensor(mean, dtype=self.x_mean.dtype, device=self.x_mean.device)
        std_t  = torch.as_tensor(std,  dtype=self.x_std.dtype,  device=self.x_std.device).clamp_min(eps)
        self.x_mean.copy_(mean_t)
        self.x_std.copy_(std_t)
        self.normalize_inputs = True 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, num_inputs)
        returns: (batch_size, num_outputs)
        """
        # x: (batch_size, seq_len, num_inputs)
        if self.normalize_inputs:
            x = (x - self.x_mean) / self.x_std
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        # Pass through final linear + activation
        return self.linear(self.activation(last_output))  


def _orthogonal_init(m: nn.Module):
    if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

