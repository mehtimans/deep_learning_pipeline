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
from typing import List

from deep_learning_course.utils import get_activation
from deep_learning_course.models import MLPNetwork


class CNNNetwork(nn.Module):
    """
    Generic configurable CNN + MLP classifier.

    conv_blocks : List[dict]
        Defines the CNN architecture. Each dictionary represents one
        convolutional block executed sequentially.
        Block fields:
        - out_channels: number of output feature maps
        - num_convs: number of Conv2D layers applied sequentially
        - kernel_size, stride, padding: convolution parameters
        - pool: optional max-pooling configuration applied after the block
                {"size": int, "stride": int, "padding": int}
                or None
        - batch_normalization: whether BatchNorm2d is added after each conv
    """
    def __init__(self,
                input_channels: int = 1,
                input_height: int = 28,
                input_width: int = 28,
                num_outputs: int = 10,
                conv_blocks=None,
                cnn_activation: str = "relu",
                mlp_network_hidden_dims: List[int] = [128, 64],
                mlp_activation: str = "relu",
                **kwargs):
        
        if kwargs:
                print("CNNNetwork.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(CNNNetwork, self).__init__()

        if conv_blocks is None:
            raise ValueError("conv_blocks must be provided")
        
        # normalization buffers
        self.normalize_inputs = False
        self.register_buffer("x_mean", torch.zeros(1, input_channels, 1, 1))
        self.register_buffer("x_std",  torch.ones(1, input_channels, 1, 1))

        # CNN Network
        # activation = get_activation(activation)
        input_channels_current = input_channels
        current_height = input_height
        current_width = input_width
        net_layers = []

        for block in conv_blocks:

            output_channels = block["out_channels"]
            number_of_convs = block["num_convs"]
            kernel_size = block["kernel_size"]
            stride = block["stride"]
            padding = block["padding"]
            pool_config = block.get("pool", None)
            batch_norm = block.get("batch_normalization", False)

            # multiple conv layers per block
            for _ in range(number_of_convs):

                net_layers.append(
                    nn.Conv2d(
                        input_channels_current,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
                # optional BatchNorm
                if batch_norm:
                    net_layers.append(nn.BatchNorm2d(output_channels))

                net_layers.append(get_activation(cnn_activation))

                # update spatial dimensions
                current_height = (current_height + 2 * padding - kernel_size) // stride + 1
                current_width  = (current_width  + 2 * padding - kernel_size) // stride + 1

                input_channels_current = output_channels

            # pooling
            if pool_config is not None:

                pool_size = pool_config["size"]
                pool_stride = pool_config["stride"]
                pool_padding = pool_config["padding"]
            
                net_layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_size,
                        stride=pool_stride,
                        padding=pool_padding
                    )
                )

                # update spatial dimensions
                current_height = (current_height + 2 * pool_padding - pool_size) // pool_stride + 1
                current_width  = (current_width  + 2 * pool_padding - pool_size) // pool_stride + 1

        self.convolutional_net = nn.Sequential(*net_layers)

        # compute flattened dimension
        flattened_feature_size = input_channels_current * current_height * current_width

        print(f"CNN Network Output size: [1, {input_channels_current}, {current_height}, {current_width}]")
        print(f"Flattened CNN feature size: {flattened_feature_size}")
        print(f"CNN Network Structure: {self.convolutional_net}")

        # Fully connected layers
        self.mlp_net = MLPNetwork(
                        num_inputs=flattened_feature_size, 
                        num_outputs=num_outputs,
                        network_hidden_dims = mlp_network_hidden_dims,
                        activation = mlp_activation)

        # initialize weights
        self.apply(_orthogonal_init)

        self.CNN_flag = True

    @torch.no_grad()
    def set_normalization(self, mean, std, eps: float = 1e-6):
        """Store per-feature mean/std (from TRAIN split). Enables normalization in forward()."""
        mean_t = torch.as_tensor(mean, dtype=self.x_mean.dtype, device=self.x_mean.device)
        std_t  = torch.as_tensor(std,  dtype=self.x_std.dtype,  device=self.x_std.device).clamp_min(eps)

        if mean_t.ndim == 1:
            mean_t = mean_t.view(1, -1, 1, 1)
            std_t  = std_t.view(1, -1, 1, 1)

        self.x_mean.copy_(mean_t)
        self.x_std.copy_(std_t)
        self.normalize_inputs = True 

    def forward(self, x):
        if self.normalize_inputs:
            x = (x - self.x_mean) / self.x_std

        x = self.convolutional_net(x)  # CNN feature extraction
        x = x.view(x.size(0), -1)  # flatten

        features = self.mlp_net.features(x)
        logits = self.mlp_net.classifier(features)
        weights = self.mlp_net.classifier.weight

        return logits, features, weights


class CNNNetworkIAM(nn.Module):
    def __init__(self,
                input_channels: int = 1,
                input_height: int = 28,
                input_width: int = 28,
                num_outputs: int = 10,
                conv_blocks=None,
                cnn_activation: str = "relu",
                mlp_network_hidden_dims: List[int] = [128, 64],
                mlp_activation: str = "relu",
                **kwargs):
        
        if kwargs:
                print("CNNNetworkIAM.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(CNNNetworkIAM, self).__init__()

        if conv_blocks is None:
            raise ValueError("conv_blocks must be provided")
        
        # normalization buffers
        self.normalize_inputs = False
        self.register_buffer("x_mean", torch.zeros(1, input_channels, 1, 1))
        self.register_buffer("x_std",  torch.ones(1, input_channels, 1, 1))

        # CNN Network
        # activation = get_activation(activation)
        input_channels_current = input_channels
        current_height = input_height
        current_width = input_width
        net_layers = []

        for block in conv_blocks:

            output_channels = block["out_channels"]
            number_of_convs = block["num_convs"]
            kernel_size = block["kernel_size"]
            stride = block["stride"]
            padding = block["padding"]
            pool_config = block.get("pool", None)
            batch_norm = block.get("batch_normalization", False)

            # multiple conv layers per block
            for _ in range(number_of_convs):

                net_layers.append(
                    nn.Conv2d(
                        input_channels_current,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
                # optional BatchNorm
                if batch_norm:
                    net_layers.append(nn.BatchNorm2d(output_channels))

                net_layers.append(get_activation(cnn_activation))

                # update spatial dimensions
                current_height = (current_height + 2 * padding - kernel_size) // stride + 1
                current_width  = (current_width  + 2 * padding - kernel_size) // stride + 1

                input_channels_current = output_channels

            # pooling
            if pool_config is not None:

                pool_size = pool_config["size"]
                pool_stride = pool_config["stride"]
                pool_padding = pool_config["padding"]
            
                net_layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_size,
                        stride=pool_stride,
                        padding=pool_padding
                    )
                )

                # update spatial dimensions
                current_height = (current_height + 2 * pool_padding - pool_size) // pool_stride + 1
                current_width  = (current_width  + 2 * pool_padding - pool_size) // pool_stride + 1

        self.convolutional_net = nn.Sequential(*net_layers)

        # compute flattened dimension
        flattened_feature_size = input_channels_current * current_height * current_width

        print(f"CNN Network Output size: [1, {input_channels_current}, {current_height}, {current_width}]")
        print(f"Flattened CNN feature size: {flattened_feature_size}")
        print(f"CNN Network Structure: {self.convolutional_net}")

        # Fully connected layers
        self.mlp_net = MLPNetwork(
                        num_inputs=flattened_feature_size, 
                        num_outputs=num_outputs,
                        network_hidden_dims = mlp_network_hidden_dims,
                        activation = mlp_activation)

        # initialize weights
        self.apply(_orthogonal_init)

        self.iam_loss = False
        self.angular_margin_loss = True

    @torch.no_grad()
    def set_normalization(self, mean, std, eps: float = 1e-6):
        """Store per-feature mean/std (from TRAIN split). Enables normalization in forward()."""
        mean_t = torch.as_tensor(mean, dtype=self.x_mean.dtype, device=self.x_mean.device)
        std_t  = torch.as_tensor(std,  dtype=self.x_std.dtype,  device=self.x_std.device).clamp_min(eps)

        if mean_t.ndim == 1:
            mean_t = mean_t.view(1, -1, 1, 1)
            std_t  = std_t.view(1, -1, 1, 1)

        self.x_mean.copy_(mean_t)
        self.x_std.copy_(std_t)
        self.normalize_inputs = True 

    def forward(self, x):
        if self.normalize_inputs:
            x = (x - self.x_mean) / self.x_std

        x = self.convolutional_net(x)  # CNN feature extraction
        x = x.view(x.size(0), -1)  # flatten


        features = self.mlp_net.features(x)
        logits = self.mlp_net.classifier(features)
        weights = self.mlp_net.classifier.weight

        return logits, features, weights
    
def _orthogonal_init(m: nn.Module):
    # Apply orthogonal initialization to convolution layers
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
