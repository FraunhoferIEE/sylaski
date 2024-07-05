from collections import OrderedDict
from math import log2
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from syndatagenerators.models.progressive_gan.model import ProGANGP, ProCritic, ProGenerator
from syndatagenerators.models.utils.plot_functions import plot_samples
from syndatagenerators.models.utils.losses import WGANGPLoss


class TrainerProGAN(pl.LightningModule):
    """
    Trainer class of the Progressive GAN. Contains necessary functions for the training
    and fading in process of new layers.
    """

    def __init__(self, train_params, dis_params, gen_params):
        """
        Args:
            train_params: dictionary of training parameters.
            dis_params: dictionary of discriminator parameters.
            gen_params: dictionary of generator parameters.
        """
        super().__init__()

        self.model_cls = ProGANGP
        self.gen_cls = ProGenerator
        self.dis_cls = ProCritic
        self.target_len = train_params['target_len']
        self.feature_dim = train_params['feature_dim']

        # train parameters
        self.batch_size = train_params['batch_size']
        self.lambda_gp = train_params['lambda_gp']
        self.n_critic = train_params['n_critic']
        self.epochs = train_params['epochs']
        self.lr = train_params['lr']
        self.nb_fade_in_epochs = train_params['nb_fade_in_epochs']
        self.schedule = train_params['schedule']
        self.name = train_params['name']

        self.optimizer = torch.optim.RMSprop
        self.loss = WGANGPLoss(lambda_gp=self.lambda_gp)

        # pretrain schedule
        self.pretrain_schedule = []
        for k in train_params['schedule']:
            self.pretrain_schedule.append((k, k + self.nb_fade_in_epochs))  # schedule for the adding of new layers
        self.nb_stage = len(train_params['schedule']) if train_params[
            'schedule'] else 0  # number of upcoming fade in stages
        self.residual_factor = 0.0
        self.depth = 0

        # network params
        self.kernel_size_dis = dis_params['kernel_size']
        self.channel_dim_dis = dis_params['channel_dim']

        self.kernel_size_gen = gen_params['kernel_size']
        self.channel_dim_gen = gen_params['channel_dim']

        self.save_hyperparameters()

        self.generator = self.gen_cls(target_len=self.target_len, nb_features=self.feature_dim,
                                      channel_nb=self.channel_dim_gen, kernel_size=self.kernel_size_gen,
                                      residual_factor=self.residual_factor)

        self.discriminator = self.dis_cls(target_len=self.target_len, nb_features=self.feature_dim,
                                          channel_nb=self.channel_dim_dis, kernel_size=self.kernel_size_dis,
                                          residual_factor=self.residual_factor)

        self.models = {'generator': self.generator,
                       'discriminator': self.discriminator}

        self.validation_z = torch.randn(10, self.feature_dim, self.target_len)

    def forward(self, z: torch.Tensor, depth: int, residual: bool, **kwargs):
        """
        Forward pass of the model. Returns the output of the generator.
        Args:
            z: latent vector of shape [n_samples, feature_dim, 2**(n+depth)]
            depth: depth (number of conv blocks) used in the generator.
            residual: boolean stating whether model is in residual phase (i.e. in a fading in process)
        """
        return self.generator(z, depth=depth, residual=residual)

    def training_step(self, batch, batch_idx, optimizer_idx, **kwargs):

        # check if model is in residual phase
        residual = self._residual(self.current_epoch)
        # check if depth of model has to be increased.
        self._increase_depth(self.current_epoch)
        depth = self.depth
        batch_size = batch.size(0)

        # generate batch of synthetic samples
        z = torch.randn(batch_size, self.feature_dim, self.target_len).float().to(batch)
        x_gen = self(z, depth, residual)

        # reduce x_batch in length according to depth of model
        reduce_factor = int(log2(self.target_len)) - int(log2(x_gen.size(2)))
        batch = F.avg_pool1d(batch, kernel_size=2 ** reduce_factor)

        # train generator
        if optimizer_idx == 0:
            g_loss = self.loss(self.discriminator, batch, x_gen, depth, residual, step=optimizer_idx)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log('generator loss', g_loss, on_epoch=True)
            return output

        # train critic
        if optimizer_idx == 1:
            d_loss = self.loss(self.discriminator, batch, x_gen, depth, residual, step=optimizer_idx)
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log('critic loss', d_loss, on_epoch=True)
            return output

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        gen_opt = self.optimizer(self.generator.parameters(), lr=self.lr)
        dis_opt = self.optimizer(self.discriminator.parameters(), lr=self.lr)
        n_critic = self.n_critic
        return (
            {'optimizer': gen_opt, 'frequency': 1},
            {'optimizer': dis_opt, 'frequency': n_critic},
        )

    def on_validation_epoch_end(self):
        pass
        # residual = self._residual(self.current_epoch)
        # print(f'Depth on epoch {self.current_epoch}: {self.depth}, residual: {residual}')
        #
        # # check if depth of model has to be increased.
        # self._increase_depth(self.current_epoch)
        # depth = self.depth
        #
        # x_gen = self(self.validation_z, depth, residual)
        #
        # fig_samples = plot_samples(x_gen, title=f'Generated samples epoch {self.current_epoch}, depth {depth}',
        #                            y_scale=None)
        # self.logger.experiment.add_figure("Generated samples", fig_samples, self.current_epoch)

    def _residual(self, epoch):
        """
        determines whether residual in networks is set to True or False (adding new layer or not)
        """
        if self.nb_stage >= 0:
            if len(self.pretrain_schedule) > 0:
                self.start_epoch_test = self.pretrain_schedule[0][0]
                self.end_epoch_test = self.pretrain_schedule[0][1]
                if (
                        self.end_epoch_test
                        > epoch
                        > self.start_epoch_test
                ):
                    self.start_epoch = self.pretrain_schedule[0][0]
                    self.end_epoch = self.pretrain_schedule[0][1]
                    self.pretrain_schedule.pop(0)

        try:
            if self.end_epoch >= epoch >= self.start_epoch:
                residual_factor = self._linear_interpolation(self.start_epoch, self.end_epoch, epoch)
                self.generator.residual_factor = residual_factor
                self.discriminator.residual_factor = residual_factor

                return True
            else:
                return False

        except Exception:
            return False

    def _increase_depth(self, epoch):
        if self.nb_stage > 0:
            self.update_epoch = self.schedule[0]
            if epoch > self.update_epoch:
                self.depth += 1
                self.nb_stage -= 1
                self.schedule.pop(0)

    def _linear_interpolation(self, start_epoch, end_epoch, epoch):
        assert end_epoch > start_epoch
        return (epoch - start_epoch) / (end_epoch - start_epoch)
