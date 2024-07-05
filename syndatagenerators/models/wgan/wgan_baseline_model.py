from collections import OrderedDict

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.utils import spectral_norm

from syndatagenerators.models.utils.plot_functions import plot_samples
from syndatagenerators.models.utils.losses import WGANGPLoss

"""
Baseline model implementation of a Wasserstein GAN model. Uses 1D-convolutional layers in 
critic and generator. 
"""


class ConvBlock(nn.Module):
    """
    Convolutional block used in critic and generator. Consists of a 1D convolution, followed by LeakyReLu activation.
    Additionally, spectral normalization is used for normalizing the weights.
    It computes dilation and padding s.t output size stays the same. as input.
    """

    def __init__(self, in_channels, out_channels, kernel_size, relu_slope, normalization='spectral'):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: length of kernel in 1D convolution
            relu_slope: slope of LeakyReLU activation
            normalization: normalization used in conv layers. Options: {'spectral', 'batch'}
        """
        super().__init__()
        dilation, padding = self._set_params(kernel_size)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.spectral_conv = spectral_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding), n_power_iterations=10)

        self.leakyrelu = nn.LeakyReLU(relu_slope)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.normalization = normalization

    @staticmethod
    def _set_params(kernel_size):
        """
        Computes dilation and padding parameter given the kernel size
        The dilation and padding parameter are computed such as
        an input sequence to a 1d convolution does not change in length.
        Note that here, the stride in the convolution is always set to 1.
        """
        if kernel_size % 2 == 1:  # odd kernel size --> dilation=1, adjust padding
            dilation = 1
            padding = int((kernel_size - 1) / 2)
        else:
            dilation = 2
            padding = int(kernel_size - 1)

        return dilation, padding

    def forward(self, x):
        """
        Forward pass of the conv block.
        Args:
            x: torch.Tensor of shape (batch_size, nb_in_features, seq_len)
        Returns:
            out: torch.Tensor of shape (batch_size, nb_out_features, seq_len)
        """
        if self.normalization == 'spectral':
            x = self.spectral_conv(x)
        elif self.normalization == 'batch':
            x = self.conv(x)
            x = self.batch_norm(x)
        out = self.leakyrelu(x)

        return out


class Critic(nn.Module):
    """
    Critic model of the (W)GAN.
    Outputs a score (scalar value) for each sample, yielding the "realness" of "fakeness" of it.

    """

    def __init__(self, input_shape: tuple = (1, 48), kernel_size: int = 7, channel_nb: int = 32,
                 relu_slope: float = 0.2, nb_layers: int = 4):
        """
        Args:
            input_shape: tuple of shape (n_dim, seq_len) describing shape of input samples
            kernel_size: length of kernel in 1D conv layers
            channel_nb: number of (output) channels in convolutional layers
            relu_slope: slope of LeakyReLU activation within layers

        """
        super(Critic, self).__init__()
        self.input_shape = input_shape
        self.nb_features = input_shape[0]
        self.relu_slope = relu_slope
        self.kernel_size = kernel_size
        self.channel_nb = channel_nb
        self.nb_layers = nb_layers

        self.initial_block = nn.Sequential(
            ConvBlock(self.nb_features, self.channel_nb, self.kernel_size, self.relu_slope),
            nn.MaxPool1d(2)
        )
        self.block_list = []
        for i in range(nb_layers):
            self.block_list.append(ConvBlock(self.channel_nb, self.channel_nb, self.kernel_size, self.relu_slope))

        self.model = nn.Sequential(
            ConvBlock(self.nb_features, self.channel_nb, self.kernel_size, self.relu_slope),
            nn.MaxPool1d(2),

            ConvBlock(self.channel_nb, self.channel_nb, self.kernel_size, self.relu_slope),
            nn.MaxPool1d(2),

            ConvBlock(self.channel_nb, self.channel_nb, self.kernel_size, self.relu_slope),
            nn.MaxPool1d(2),

            ConvBlock(self.channel_nb, self.channel_nb, self.kernel_size, self.relu_slope),
            nn.MaxPool1d(2)
        )

        ds_size = int(np.ceil(self.input_shape[1] / 2 ** self.nb_layers))  # sequence length reduction (after 4 times max pooling)

        self.fc = nn.Sequential(
            nn.Linear(self.channel_nb * ds_size, 64),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Linear(64, 1)  # output linear layer to 1 dim output

    def forward(self, x: torch.Tensor):

        """
        Forward pass of Critic.
        Args:
            x: torch.Tensor of shape [batch_size, feature_dim, seq_len]
        Returns:
            validity: torch.Tensor of shape [batch_size, 1]
        """
        if x.dim() == 2:
            x = x.view(x.size(0), 1, -1)  # reshape to 3 d (only necessary for 2 dim inputs with 1 feature)

        out = self.model(x)
        out = out.view(out.size(0), -1)
        feature = self.fc(out)
        validity = self.fc2(feature)
        return validity


class Generator(nn.Module):
    """
    Generator model of (W)GAN. Ued to generate synthetic samples of the same shape than the original ones.
    """
    def __init__(self, input_shape: tuple = (1,48), latent_dim: int = 100, kernel_size: int = 3, channel_nb: int = 32,
                 relu_slope: float = 0.2):
        super(Generator, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.nb_features = input_shape[0]
        self.kernel_size = kernel_size
        self.channel_nb = channel_nb
        self.relu_slope = relu_slope
        self.init_size = int(np.ceil(self.input_shape[1]/4))  # for upsampling

        self.fc = nn.Linear(self.latent_dim, self.channel_nb * self.init_size)

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(self.channel_nb, self.channel_nb, self.kernel_size, self.relu_slope),
            nn.Upsample(scale_factor=2),
            ConvBlock(self.channel_nb, self.channel_nb, self.kernel_size, self.relu_slope),
            ConvBlock(self.channel_nb, self.channel_nb, self.kernel_size, self.relu_slope)
        )

        self.last_block = spectral_norm(
            nn.Conv1d(
                in_channels=self.channel_nb,
                out_channels=self.nb_features,
                kernel_size=1), n_power_iterations=10)

    def forward(self, z):
        """
        Forward pass of the Generator.
        Args:
            z: torch.Tensor of shpe [n_samples, latent_dim]. Latent noise vector.
        Returns:
            x: torch.Tensor of shape [n_samples, feature_dim, seq_len].
        """
        out = self.fc(z)
        out = out.view(out.size(0), self.channel_nb, self.init_size)
        pre_x = self.conv_blocks(out)
        x = self.last_block(pre_x)
        return x


class BaselineWGAN(pl.LightningModule):
    def __init__(self, gen_cls, dis_cls, args):
        super(BaselineWGAN, self).__init__()
        self.name = args.name
        self.input_shape = args.input_shape
        self.latent_dim = args.latent_dim
        self.kernel_size_dis = args.kernel_size_dis
        self.kernel_size_gen = args.kernel_size_gen
        self.channel_nb_dis = args.channel_nb_dis
        self.channel_nb_gen = args.channel_nb_gen
        if args.optimizer == 'RMSProp':
            self.optimizer = torch.optim.RMSprop
        elif args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam
        self.lr_dis = args.lr_dis
        self.lr_gen = args.lr_gen
        self.lambda_gp = args.lambda_gp
        self.loss = WGANGPLoss(self.lambda_gp)
        self.n_critic = args.n_critic
        self.save_hyperparameters()
        self.automatic_optimization = True

        self.generator = gen_cls(input_shape=self.input_shape, latent_dim=self.latent_dim,
                                 kernel_size=self.kernel_size_gen,channel_nb=self.channel_nb_gen)

        self.discriminator = dis_cls(input_shape=self.input_shape, kernel_size=self.kernel_size_dis,
                                     channel_nb=self.channel_nb_dis)

        self.models = {'generator': self.generator,
                       'discriminator': self.discriminator}

        # two fix latent vectors for validation during training
        self.validation_z_1 = torch.randn(10, self.latent_dim)
        self.validation_z_2 = torch.randn(10, self.latent_dim)

    def forward(self, z: torch.Tensor):
        return self.generator(z)

    def classifier_loss(self, x_real, x_gen):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx, log_samples=False, **kwargs):

        z = torch.randn(batch.size(0), self.latent_dim)
        z.type_as(batch)

        # train generator
        if optimizer_idx == 0:
            if log_samples:
                self.gen_samples = self(z)
                sample = self.gen_samples[:6]
                fig = plot_samples(sample, figsize=(5, 18), title='generated samples')
                self.logger.experiment.add_figure("Generated sequences", fig, 0)

            g_loss = self.loss(self.discriminator, batch, self(z), step='generator')
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output
        # train critic
        if optimizer_idx == 1:
            d_loss = self.loss(self.discriminator, batch, self(z), step='discriminator')
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log('critic loss', d_loss, on_epoch=True)
            return output

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        gen_opt = self.optimizer(self.generator.parameters(), lr=self.lr_gen)
        dis_opt = self.optimizer(self.discriminator.parameters(), lr=self.lr_dis)
        n_critic = self.n_critic
        return (
            {'optimizer': gen_opt, 'frequency': 1},
            {'optimizer': dis_opt, 'frequency': n_critic},
        )

    def on_epoch_end(self):
        # plot samples
        z_1 = self.validation_z_1
        z_2 = self.validation_z_2
        sample_series_1 = self(z_1).detach()
        sample_series_2 = self(z_2).detach()
        fig_1 = plot_samples(sample_series_1, title='Generated sequences')
        fig_2 = plot_samples(sample_series_2, title='Generated sequences')
        self.logger.experiment.add_figure("Generated sequences, latent vector 1", fig_1, self.current_epoch)
        self.logger.experiment.add_figure("Generated sequences, latent vector 2", fig_2, self.current_epoch)


