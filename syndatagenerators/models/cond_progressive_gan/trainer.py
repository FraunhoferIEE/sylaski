from math import log2
from collections import OrderedDict

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from syndatagenerators.models.progressive_gan.trainer import TrainerProGAN
from syndatagenerators.models.cond_progressive_gan.networks import CPGenerator, CPDiscriminator


class TrainerCPGAN(TrainerProGAN, pl.LightningModule):
    """
    Trainer class of the Conditional Progressive GAN. Contains necessary functions for the training
    and fading in process of new layers.
    """

    def __init__(self, train_params: dict, dis_params: dict, gen_params: dict, nb_classes: int):
        """
        Args:
            train_params: dictionary of training parameters.
            dis_params: dictionary of discriminator parameters.
            gen_params: dictionary of generator parameters.
            nb_classes: number of different classes used for conditioning.
        """
        super(TrainerCPGAN, self).__init__(train_params=train_params, dis_params=dis_params, gen_params=gen_params)

        self.gen_cls = CPGenerator
        self.dis_cls = CPDiscriminator
        self.nb_classes = nb_classes
        self.nb_conditions = train_params['nb_labels']

        self.generator = self.gen_cls(target_len=self.target_len, nb_features=self.feature_dim,
                                      nb_classes=self.nb_classes, nb_conditions=self.nb_conditions,
                                      channel_nb=self.channel_dim_gen, kernel_size=self.kernel_size_gen,
                                      residual_factor=self.residual_factor)

        self.discriminator = self.dis_cls(target_len=self.target_len, nb_features=self.feature_dim,
                                          nb_classes=self.nb_classes, nb_conditions=self.nb_conditions,
                                          channel_nb=self.channel_dim_dis, kernel_size=self.kernel_size_dis,
                                          residual_factor=self.residual_factor)

    def forward(self, z: torch.Tensor, id_hs: torch.Tensor, depth: int, residual: bool):
        """
        Forward pass of the conditional Progressive GAN. Returns the output iof the generator.
        Args:
            z: latent vector of shape [n_samples, feature_dim, 2**(n+depth)]
            id_hs: tensor of the household ID's
            depth: depth (number of conv blocks) used in the generator.
            residual: boolean stating whether model is in residual phase (i.e. in a fading in process)
        """
        return self.generator(z, id_hs, depth, residual)

    def training_step(self, batch, batch_idx, optimizer_idx, **kwargs):

        x_batch, label = batch

        # check if model is in residual phase
        residual = self._residual(self.current_epoch)
        # check if depth of model has to be increased.
        self._increase_depth(self.current_epoch)
        depth = self.depth
        batch_size = x_batch.size(0)

        assert label.dtype == torch.int32, "Input labels need to be of integer type torch.int32"

        # generate batch of synthetic samples
        z = torch.randn(batch_size, self.feature_dim, self.target_len, dtype=torch.float32).to(self.device)
        x_gen = self(z, label, depth, residual)

        # reduce x_batch in length according to depth of model
        reduce_factor = int(log2(self.target_len)) - int(log2(x_gen.size(2)))
        x_batch = F.avg_pool1d(x_batch, kernel_size=2 ** reduce_factor)

        # train generator
        if optimizer_idx == 0:
            g_loss = self.loss(self.discriminator, x_batch, x_gen, label, residual, step=optimizer_idx)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log('generator loss', g_loss, on_epoch=True)
            return output

        # train critic
        if optimizer_idx == 1:
            d_loss = self.loss(self.discriminator, x_batch, x_gen, label, residual, step=optimizer_idx)
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log('critic loss', d_loss, on_epoch=True)
            return output
