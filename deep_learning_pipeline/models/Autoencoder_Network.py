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


class AutoEncoderNetwork(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 latent_dim: int,
                 encoder_hidden_dims: List[int] = [128, 64],
                 decoder_hidden_dims:  List[int] = [64, 128],
                 activation: str = "softsign",
                 **kwargs):

        if kwargs:
            print("AutoEncoderNetwork.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))

        super(AutoEncoderNetwork, self).__init__()

        # normalization buffers
        self.register_buffer("x_mean", torch.zeros(num_inputs))
        self.register_buffer("x_std", torch.ones(num_inputs))
        self.normalize_inputs = False

        # activation = get_activation(activation)
        encoder_input_dim = num_inputs
        decoder_output_dim = num_inputs

        # Encoder Network
        encoder_layers = []
        if len(encoder_hidden_dims) == 0:
            encoder_layers.append(nn.Linear(encoder_input_dim, latent_dim))

        else:
            encoder_layers.append(nn.Linear(encoder_input_dim, encoder_hidden_dims[0]))
            encoder_layers.append(get_activation(activation))
            for e in range(len(encoder_hidden_dims)):
                if e == len(encoder_hidden_dims)-1:
                    encoder_layers.append(nn.Linear(encoder_hidden_dims[e], latent_dim))
                else:
                    encoder_layers.append(nn.Linear(encoder_hidden_dims[e], encoder_hidden_dims[e + 1]))
                    encoder_layers.append(get_activation(activation))
        self.encoder_net = nn.Sequential(*encoder_layers)

        # Decoder Network
        decoder_layers = []
        if len(decoder_hidden_dims) == 0:
            decoder_layers.append(nn.Linear(latent_dim, decoder_output_dim))

        else:
            decoder_layers.append(nn.Linear(latent_dim, decoder_hidden_dims[0]))
            decoder_layers.append(get_activation(activation))
            for d in range(len(decoder_hidden_dims)):
                if d == len(decoder_hidden_dims)-1:
                    decoder_layers.append(nn.Linear(decoder_hidden_dims[d], decoder_output_dim))
                else:
                    decoder_layers.append(nn.Linear(decoder_hidden_dims[d], decoder_hidden_dims[d + 1]))
                    decoder_layers.append(get_activation(activation))
        self.decoder_net = nn.Sequential(*decoder_layers)


        self.apply(_orthogonal_init)
        nn.init.orthogonal_(self.encoder_net[-1].weight, gain=0.01)
        nn.init.zeros_(self.encoder_net[-1].bias)
        nn.init.orthogonal_(self.decoder_net[-1].weight, gain=0.01)
        nn.init.zeros_(self.decoder_net[-1].bias)

        print(f"Encoder Structure: {self.encoder_net}")
        print(f"Decoder Structure: {self.decoder_net}")

    @torch.no_grad()
    def set_normalization(self, mean, std, eps: float = 1e-6):
        mean_t = torch.as_tensor(mean, dtype=self.x_mean.dtype, device=self.x_mean.device)
        std_t = torch.as_tensor(std, dtype=self.x_std.dtype, device=self.x_std.device).clamp_min(eps)

        self.x_mean.copy_(mean_t)
        self.x_std.copy_(std_t)
        self.normalize_inputs = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_inputs:
            x = (x - self.x_mean) / self.x_std
        return self.encoder_net(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


def _orthogonal_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
