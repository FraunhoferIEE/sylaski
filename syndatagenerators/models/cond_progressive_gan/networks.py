from math import log2

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from syndatagenerators.models.progressive_gan.networks import ProCritic, ProGenerator, ConvBlock


class CPGenerator(ProGenerator):
    """
   Conditional version of the ProGenerator. Uses as additional input ID's which are given into an embedding and then
   concatenated to the input.
   """

    def __init__(self, target_len: int, nb_features: int, nb_classes: int, nb_conditions: int = 1, channel_nb: int = 32,
                 kernel_size: int = 3, relu_slope: float = 0.2, residual_factor: float = 0.0):
        super().__init__(target_len=target_len, nb_features=nb_features, channel_nb=channel_nb, kernel_size=kernel_size,
                         relu_slope=relu_slope, residual_factor=residual_factor)
        self.nb_conditions = nb_conditions
        self.nb_classes = nb_classes

        # embedding for household ID's
        self.embedding = nn.Embedding(nb_classes, self.target_len)

        self.initial_block = ConvBlock(
            in_channel=nb_features + nb_conditions,
            out_channel=channel_nb,
            kernel_size=kernel_size,
            relu_slope=relu_slope
        )

    def forward(self, z: torch.Tensor, id_h: torch.Tensor, depth: int = None,
                residual: bool = False):
        """
       Forward pass of the conditional generator of the Conditional Progressive GAN.
       Args:
          z: torch.Tensor of shape [batch_size, feature-sim, target_len]
          id_h: id of the respective household
          depth: current depth of the model. Has to be in {0, ..., nb_steps -1 }
          residual: boolean stating whether residual phase (fading in process) is active.
       """

        # depth fo reaching target length
        if depth is None:
            depth = self.nb_step - 1

        assert z.dim() == 3, "input must be three dimensional"
        assert (
                z.size(2) == self.target_len
        ), "third dimension of input must be equal to target_len"
        assert depth <= self.nb_step - 1, "depth is too high"

        embedded_id = self.embedding(id_h)
        z_joint = torch.cat([z, embedded_id], 1)  # concatenate id with input
        reduced_z = F.avg_pool1d(
            z_joint, kernel_size=2 ** (self.nb_step - 1)
        )  # has length 8 (i.e. shape [batch_size, nb_features, 8]

        y = self.initial_block(reduced_z)
        for idx, l in enumerate(self.block_list[:depth]):
            # upsampling
            y = F.interpolate(y, scale_factor=2, mode="nearest")  # output length: 2**(3+l)
            previous_y = y  # output before (last) layer in block list
            y = l(y)  # output after layer
            last_idx = idx  # idx of last layer in blocklist

        if residual and depth > 0:
            l_skip = self.skip_block_list[last_idx]
            y = self.residual_factor * self.last_block(y) + (
                    1 - self.residual_factor
            ) * self.last_block(l_skip(previous_y))

        else:
            y = self.last_block(y)

        return y


class CPDiscriminator(ProCritic):
    """
    Conditional version of the Progressive GAN Critic.
    """

    def __init__(self, target_len: int, nb_features: int, nb_classes: int, nb_conditions: int = 1, channel_nb: int = 32,
                 kernel_size: int = 3, relu_slope: float = 0.2, residual_factor: float = 0.0):
        super().__init__(target_len=target_len, nb_features=nb_features, channel_nb=channel_nb, kernel_size=kernel_size,
                         relu_slope=relu_slope, residual_factor=residual_factor)
        self.nb_classes = nb_classes
        self.nb_conditions = nb_conditions

        self.embedding = nn.Embedding(nb_classes, self.target_len)

        self.initial_block = nn.Sequential(
            spectral_norm(
                nn.Conv1d(
                    in_channels=nb_features + nb_conditions,
                    out_channels=channel_nb,
                    kernel_size=1
                )
            ),
            nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor, id_h: torch.Tensor, residual: bool = False):

        """
           Forward pass of the conditional critic of the Conditional Progressive GAN.
           Args:
              x: torch.Tensor of shape [batch_size, feature-sim, target_len]
              id_h: id of the respective household
              depth: current depth of the model. Has to be in {0, ..., nb_steps -1 }
              residual: boolean stating whether residual phase (fading in process) is active.
        """
        assert x.dim() == 3, "input must be three dimensional"
        assert (
                x.size(2) >= 8
        ), "third dimension of input must be greater or equal than 8"
        assert (
                log2(x.size(2)) % 1 == 0
        ), "input length must be an integer that is a power of 2."

        reduce_factor = int(log2(self.target_len)) - int(log2(x.size(2)))

        embedded_id = self.embedding(id_h)
        reduced_id = F.avg_pool1d(embedded_id, kernel_size=2 ** reduce_factor)  # reduce embedded id to same length

        x_joint = torch.cat([x, reduced_id], 1)  # concatenate input noise with time

        if residual:
            pre_x = F.avg_pool1d(x_joint, kernel_size=2)

            pre_x = self.initial_block(pre_x)

        x = self.initial_block(x_joint)

        for idx, l in enumerate(self.block_list[reduce_factor:]):
            x = l(x)
            x = F.avg_pool1d(x, kernel_size=2)
            if idx == 0:
                if residual:
                    x = (
                            self.residual_factor * x
                            + (1 - self.residual_factor) * pre_x
                    )

        x = self.last_block(x)
        x = self.fc(x.squeeze(1))

        return x
