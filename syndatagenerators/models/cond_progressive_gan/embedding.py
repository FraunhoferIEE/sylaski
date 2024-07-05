import numpy as np
import torch
from torch import nn


class EmbeddingNN(nn.Module):
    """
    Neural network that uses 1D convolution to embed household ID's to a given length. Can be treained via contrastive
    learning.
    """

    def __init__(self, input_len: int = 256, num_in_channels: int = 1, num_hidden_channels: int = 128,
                 len_embedding: int = 32, len_last_kernel: int = 20):
        super().__init__()
        self.input_len = input_len
        self.num_in_channels = num_in_channels
        self.num_hidden_channels = num_hidden_channels
        self.len_embedding = len_embedding
        self.len_last_kernel = len_last_kernel

        assert input_len // 2**4 > 1, "input length must be long enough for number of downsizing layers"

        self.mapping = nn.Sequential(
            nn.Conv1d(num_in_channels, num_hidden_channels,
                      kernel_size=3, stride=2),
            nn.BatchNorm1d(num_hidden_channels),
            nn.LeakyReLU(),
            nn.Conv1d(num_hidden_channels, num_hidden_channels,
                      kernel_size=3, stride=2),
            nn.BatchNorm1d(num_hidden_channels),
            nn.LeakyReLU(),
            nn.Conv1d(num_hidden_channels, num_hidden_channels,
                      kernel_size=3, stride=2),
            nn.BatchNorm1d(num_hidden_channels),
            nn.LeakyReLU(),
            nn.Conv1d(num_hidden_channels, num_hidden_channels,
                      kernel_size=3, stride=2),
            nn.BatchNorm1d(num_hidden_channels),
            nn.LeakyReLU(),
            nn.Conv1d(num_hidden_channels,
                      num_hidden_channels, kernel_size=len_last_kernel),
            nn.BatchNorm1d(num_hidden_channels),
            nn.LeakyReLU(),
            nn.Conv1d(num_hidden_channels, len_embedding, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        out = self.mapping(x)

        return out

    def train(self, loader, n_epochs: int = 10):
        pass

