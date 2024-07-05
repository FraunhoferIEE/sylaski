from math import log2

import torch
from torch import nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional block used in critic and generator. Consists of a 1D convolution, followed by LeakyReLu activation.
    Additionally, spectral normalization is used for normalizing the weights, or optionally batch normalization.
    It computes dilation and padding s.t output size stays the same. as input.

    """

    def __init__(self, in_channel, out_channel, kernel_size: int, relu_slope: float, normalization: str = 'spectral'):
        """
        Args:
            in_channel: number of input channels
            out_channel: number of output channels
            kernel_size: length of kernel in 1D convolution
            relu_slope: slope of LeakyReLU activation
            normalization: normalization used in conv layers. Options: {'spectral', 'batch'}
        """
        super().__init__()
        dilation, padding = self._set_params(kernel_size)
        self.conv = nn.Conv1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.spectral_conv = spectral_norm(
            nn.Conv1d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding),
            n_power_iterations=10)

        self.leakyrelu = nn.LeakyReLU(relu_slope)
        self.batch_norm = nn.BatchNorm1d(out_channel)
        self.normalization = normalization
        assert self.normalization in ['spectral', 'batch'], "use either 'spectral' or 'batch' as normalization method"

    @staticmethod
    def _set_params(kernel_size):
        """Computes dilation and padding parameter given the kernel size
        The dilation and padding parameter are computed such as
        an input sequence to a 1d convolution does not change in length.
        Returns:
            Two integer for dilation and padding.
        """

        if kernel_size % 2 == 1:  # If kernel size is an odd number
            dilation = 1
            padding = int((kernel_size - 1) / 2)
        else:  # If kernel size is an even number
            dilation = 2
            padding = int(kernel_size - 1)

        return dilation, padding

    def forward(self, x_batch: torch.Tensor):
        """
        Forward pass of the conv block.
        Args:
            x_batch: torch.Tensor of shape (batch_size, nb_in_features, seq_len)
        Returns:
            out: torch.Tensor of shape (batch_size, nb_out_features, seq_len)
        """
        if self.normalization == 'spectral':
            feature = self.spectral_conv(x_batch)
        elif self.normalization == 'batch':
            feature = self.conv(x_batch)
            feature = self.batch_norm(feature)
        out = self.leakyrelu(feature)

        return out


class ProGenerator(nn.Module):
    """
    Generator of the Progressive GAN. The number of inner convolutional blocks is determined based on the target length
    of the output sequences.
    """

    def __init__(self, target_len: int, nb_features: int, channel_nb: int, kernel_size: int = 3,
                 relu_slope: float = 0.2, residual_factor: float = 0.0):
        """
        Args:
            target_len: target length of the (final) output sequence. Needs to be a power of 2, and larger/equal 8.
            nb_features: number of features of the input samples.
            channel_nb: hidden channel dimension of the convolutional blocks.
            kernel_size: kernel size in the 1D convolution.
            relu_slope: slope of leaky relu in convolutional blocks.
        """

        super().__init__()

        self.target_len = target_len
        self.nb_features = nb_features  # number of features
        self.channel_nb = channel_nb
        self.kernel_size = kernel_size
        self.relu_slope = relu_slope
        self.residual_factor = residual_factor

        assert (log2(target_len) % 1 == 0), "target len must be an integer that is a power of 2."
        assert target_len >= 8, "target len should be at least of value 8."

        self.nb_step = int(log2(target_len)) - 2  # number of step for adding layers

        # initial block
        self.initial_block = ConvBlock(
            in_channel=nb_features,  # left out +1 for time features
            out_channel=channel_nb,
            kernel_size=kernel_size,
            relu_slope=relu_slope
        )
        # last block
        self.last_block = spectral_norm(
            nn.Conv1d(
                in_channels=channel_nb,
                out_channels=nb_features,
                kernel_size=1,
            ))

        self.block_list = nn.ModuleList([])
        for stage in range(1, self.nb_step):
            self.block_list.append(
                ConvBlock(
                    in_channel=channel_nb,
                    out_channel=channel_nb,
                    kernel_size=kernel_size,
                    relu_slope=relu_slope)
            )
        # skip block list: for the fading in of new layers
        self.skip_block_list = nn.ModuleList([])
        for stage in range(1, self.nb_step):
            self.skip_block_list.append(
                nn.Conv1d(
                    in_channels=channel_nb,
                    out_channels=channel_nb,
                    kernel_size=1
                )
            )  # fully connected layer

    def forward(self, z: torch.Tensor, depth: int = None,
                residual: bool = False):
        """
        Forward pass of the generator of the Progressive GAN.
        Args:
             z: torch.Tensor of shape [batch_size, feature-sim, target_len]
             depth: current depth of the model. Has to be in {0, ..., nb_steps -1}.
             residual: boolean stating whether residual phase (fading in process) is active.
        """

        # depth for reaching target length
        if depth is None:
            depth = self.nb_step - 1

        assert z.dim() == 3, "input must be three dimensional"
        assert (
                z.size(2) == self.target_len
        ), "third dimension of input must be equal to target_len"
        assert depth <= self.nb_step - 1, "depth is too high"

        reduced_z = F.avg_pool1d(  # max pool instead of avg pool?
            z, kernel_size=2 ** (self.nb_step - 1)
        )  # Reduce x to length 8
        y = self.initial_block(reduced_z)  # output  length after first block: 8

        for idx, l in enumerate(self.block_list[:depth]):
            # upsampling
            y = F.interpolate(y, scale_factor=2, mode="nearest")  # output length: 2**(3+l)
            previous_y = y  # output before (last) layer in block list
            y = l(y)  # output after layer
            last_idx = idx  # idx of last layer in blocklist

        if residual and depth > 0:
            l_skip = self.skip_block_list[last_idx]  #
            y = self.residual_factor * self.last_block(y) + (
                    1 - self.residual_factor
            ) * self.last_block(l_skip(previous_y))

        else:
            y = self.last_block(y)

        return y


class ProCritic(nn.Module):
    """
    Critic of the Progressive GAN. The number of inner convolutional blocks is determined based on the target length
    of the output sequences.
    """

    def __init__(self, target_len: int, nb_features: int, channel_nb: int = 32, kernel_size: int = 3,
                 relu_slope: float = 0.2, residual_factor: float = 0.0):
        """
        Args:
            target_len: integer value specifying target sequence length of input samples
            nb_features: number of different features per sample.
            channel_nb: dimension of hidden layers in the convolutional blocks.
            kernel_size: length of the kernel in the 1D convolutional layers.
            relu_slope: slope of leakyRELU in the convolutional blocks.

        """
        super().__init__()
        assert target_len >= 8, "target length should be at least of value 8"
        assert (
                log2(target_len) % 1 == 0
        ), "input length must be an integer that is a power of 2."

        self.target_len = target_len
        self.nb_step = (int(log2(target_len)) - 2)  # nb of step to go from 8 to target_len
        self.channel_nb = channel_nb
        self.residual_factor = residual_factor
        self.initial_block = nn.Sequential(
            spectral_norm(
                nn.Conv1d(
                    in_channels=nb_features,
                    out_channels=channel_nb,
                    kernel_size=1,  # kernel size = 1 --> fully connected layer
                )
            ),
            nn.LeakyReLU(),
        )

        self.last_block = nn.Sequential(
            ConvBlock(
                in_channel=channel_nb,
                out_channel=channel_nb,
                kernel_size=kernel_size,
                relu_slope=relu_slope
            ),
            spectral_norm(
                nn.Conv1d(
                    in_channels=channel_nb, out_channels=1, kernel_size=1
                )
            ),
            nn.LeakyReLU(),
        )
        self.fc = spectral_norm(nn.Linear(8, 1))
        self.block_list = nn.ModuleList([])
        for stage in range(self.nb_step - 1, 0, -1):
            self.block_list.append(
                ConvBlock(
                    in_channel=channel_nb,
                    out_channel=channel_nb,
                    kernel_size=kernel_size,
                    relu_slope=relu_slope
                )
            )

    def forward(self, x: torch.Tensor, depth=None, residual: bool = False):
        """
        Computes the forward pass of the Critic.
        Arguments:
            x: tensor of shape (batch size, 1, input_length)
            depth: the depth at which the the tensor should flow.
            residual: boolean stating whether residual phase (fading in process) is active.
        """
        assert x.dim() == 3, "input must be three dimensional"
        assert (
                x.size(2) >= 8
        ), "third dimension of input must be greater or equal than 8"
        assert (
                log2(x.size(2)) % 1 == 0
        ), "input length must be an integer that is a power of 2."
        # assert (
        #       tf.size(2) == self.target_len
        # ), "length of features should be equal to target len"
        if depth is None:
            depth = self.nb_step - 1
        reduce_factor = int(log2(self.target_len)) - int(log2(x.size(2)))  # from which block in blocklist to start from
        # --> index

        if residual:
            pre_x = F.avg_pool1d(x, kernel_size=2)  # half the output length

            pre_x = self.initial_block(pre_x)  # Output des ersten Blocks, halbierte Länge

        x = self.initial_block(x)

        for idx, l in enumerate(self.block_list[reduce_factor:]):
            x = l(x)
            x = F.avg_pool1d(x, kernel_size=2)
            if idx == 0:  # residual phase only for first block of discriminator
                if residual:
                    x = (
                            self.residual_factor * x
                            + (1 - self.residual_factor) * pre_x
                    )

        x = self.last_block(x)
        x = self.fc(x.squeeze(1))
        return x
