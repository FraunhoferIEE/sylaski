import argparse
import itertools
from typing import List

import matplotlib.pyplot as plt

from syndatagenerators.models.cycle_gan.cmd_parser import Arguments
from syndatagenerators.models.cycle_gan.time_modules import (TimeCycleDiscriminator, TimeCycleGenerator)

import torch as th
import torch.nn.functional as F

import pytorch_lightning as pl

from syndatagenerators.models.utils import data_loading

class TimeCycleGAN(pl.LightningModule):
    ''' 
    Defines cycle GAN model using residual blocks
    '''
    
    def __init__(self,
        channels,
        time_steps,
        gan_mode='unet',
        real_label=1,
        smoothed_label=1,
        fake_label=0,
        depth=2,
        residuals=3,
        sample_pool=200,
        g_lr = 0.0002,
        d_lr = 0.0002,
        cyc_lambda_A = 10,
        cyc_lambda_B = 10,
        lambda_id = 0.5,
        dilations=False,
        batch_size = 64,
        total_epochs=300,
        decay_epochs=100
    ) -> None:
        super(TimeCycleGAN, self).__init__()
        
        # saves hyper parameters into lightning framework
        self.save_hyperparameters()
        
        self.G_AB = TimeCycleGenerator(gan_mode, channels, channels, depth=depth, residuals=residuals, dilations=dilations, time_steps=time_steps)
        self.G_BA = TimeCycleGenerator(gan_mode, channels, channels, depth=depth, residuals=residuals, dilations=dilations, time_steps=time_steps)

        self.D_A = TimeCycleDiscriminator(channels, h_channels=64, in_size=time_steps)
        self.D_B = TimeCycleDiscriminator(channels, h_channels=64, in_size=time_steps)
        
        self.pool_A = data_loading.SamplePool(sample_pool)
        self.pool_B = data_loading.SamplePool(sample_pool)
        
        self.val_A = None
        self.val_B = None

        self.example_input_array = {
            'A': th.rand(2, channels, time_steps),
            'B': th.rand(2, channels, time_steps)
        }


    def forward(self, A=None, B=None):
        ''' calculates the forward pass of the network. Only computes output for given domains.
        Args:
            A (th.tensor): images of domain A
            B (th.tensor): images of domain B
        '''
        if A != None:
            fake_B = self.G_AB(A)
            if B == None:
                return fake_B
        if B != None:
            fake_A = self.G_BA(B)
            if A == None:
                return fake_A
        return fake_B, fake_A



    def adverserial_ls_loss(self, y_hat, y):
        '''Adverserial least squares loss'''
        return F.mse_loss(y_hat, y)

    def cycle_consistency_loss(self, y_hat, y):
        '''Cycle Consistency MAE loss (L1)'''
        return F.l1_loss(y_hat, y)



    def configure_optimizers(self):
        '''
        Builds 2 optimizers for both generators and both discriminators.
        Order is Generator, Discrominator
        '''
        g_lr = self.hparams.g_lr
        d_lr = self.hparams.d_lr
        total_epochs = self.hparams.total_epochs
        decay_epochs = self.hparams.decay_epochs
        burnin_epochs = total_epochs - decay_epochs
        
        # train generators and discriminators jointly
        if False:
            gen_params = itertools.chain(self.G_AB.parameters(),self.G_BA.parameters())
            opt_g = th.optim.RMSprop(gen_params, lr=g_lr)
            disc_params = itertools.chain(self.D_A.parameters(),self.D_B.parameters())
            opt_d = th.optim.RMSprop(disc_params, lr=d_lr)

            # def _lambda_rule(epoch):
            #     lr_l = 1.0 - max(0, epoch + decay_epochs - total_epochs) / float(decay_epochs + 1)
            #     return lr_l
            # scheduler_g = optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=_lambda_rule)
            # scheduler_d = optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=_lambda_rule)

            return [opt_g, opt_d] #, [scheduler_g, scheduler_d]
    
    
        self.g_ab_optimizer = th.optim.Adam(self.G_AB.parameters(), lr=self.hparams.g_lr,
                                                       betas=(0.5, 0.999))
        self.g_ba_optimizer = th.optim.Adam(self.G_BA.parameters(), lr=self.hparams.g_lr,
                                                 betas=(0.5, 0.999))
        self.d_a_optimizer = th.optim.Adam(self.D_A.parameters(), lr=self.hparams.d_lr, betas=(0.5, 0.999))
        self.d_b_optimizer = th.optim.Adam(self.D_B.parameters(), lr=self.hparams.d_lr, betas=(0.5, 0.999))

        return [self.g_ab_optimizer, self.g_ba_optimizer, self.d_a_optimizer,
                self.d_b_optimizer], []



    def training_step(self, batch, batch_idx, optimizer_idx):
        '''performs the training step of the optimization'''
        lambda_A = self.hparams.cyc_lambda_A
        lambda_B = self.hparams.cyc_lambda_B
        lambda_id = self.hparams.lambda_id
        real_label = self.hparams.real_label
        fake_label = self.hparams.fake_label
        smoothed_label = self.hparams.smoothed_label
        
        A, B = batch['A'], batch['B']
        bs = A.shape[0]
        

        if optimizer_idx == 0 or optimizer_idx == 1:  # update G_*
            # predict gans as if they were real; dont update discriminator
            data_loading.freeze_module([self.D_A, self.D_B])

            # get fake sequences and recoveries
            self.fake_B, self.fake_A, self.rec_A, self.rec_B = self.gen_all_sequences(A, B)

            pred_fake_B = self.D_B(self.fake_B)
            target = th.tensor(real_label).expand_as(pred_fake_B).type_as(A)

            loss_G_AB = self.adverserial_ls_loss(pred_fake_B, target)
            self.log('train/G_AB', loss_G_AB, prog_bar=True)

            pred_fake_A = self.D_A(self.fake_A)
            target = th.tensor(real_label).expand_as(pred_fake_A).type_as(B)

            loss_G_BA = self.adverserial_ls_loss(pred_fake_A, target)
            self.log('train/G_BA', loss_G_BA, prog_bar=True)

            # cycle consistency loss: A -> B (+ eps) -> A' => A~A'
            loss_cyc_A = self.cycle_consistency_loss(self.rec_A, A)
            self.log('train/Cycle_loss_A', loss_cyc_A, prog_bar=True)

            loss_cyc_B = self.cycle_consistency_loss(self.rec_B, B)
            self.log('train/Cycle_loss_B', loss_cyc_B, prog_bar=True)

            #trans_B, trans_A = self(B, A)
            #loss_id = self.cycle_consistency_loss(trans_A, A) * lambda_id + self.cycle_consistency_loss(trans_B, B) * lambda_id
            #self.log('train/identity_loss', loss_id, prog_bar=True)

            # combine the least squares adv. losses and cycle cons. loss; lambda is a hyperparameter
            loss_generator = loss_G_AB + loss_G_BA + loss_cyc_A * lambda_A + loss_cyc_B * lambda_B# + loss_id
            self.log('train/generator_loss', loss_generator, prog_bar=True)

            return loss_generator
        
        if optimizer_idx == 2 or optimizer_idx == 3:  #update D_*
            data_loading.unfreeze_module([self.D_A, self.D_B])

            # make predictions for real data
            target_real = th.full((bs, 1), smoothed_label).type_as(A)
            
            pred_real_A = self.D_A(A)
            loss_real_A = self.adverserial_ls_loss(pred_real_A, target_real)
            
            target_real = th.full((bs, 1), smoothed_label).type_as(B)
            pred_real_B = self.D_B(B)
            loss_real_B = self.adverserial_ls_loss(pred_real_B, target_real)

            fake_B, fake_A = self(A=A,B=B)
            
            # take old images with a probability 
            #fake_B = self.pool_B.query(fake_B).type_as(A)
            #fake_A = self.pool_A.query(fake_A).type_as(B)
            
            # make predicitions for fake data; detach to not accumulate grads in generators
            pred_fake_A = self.D_A(fake_A.detach())
            target_fake = th.tensor(fake_label).expand_as(pred_fake_A).type_as(A)
            loss_fake_A = self.adverserial_ls_loss(pred_fake_A, target_fake)

            pred_fake_B = self.D_B(fake_B.detach())
            target_fake = th.tensor(fake_label).expand_as(pred_fake_B).type_as(B)
            loss_fake_B = self.adverserial_ls_loss(pred_fake_B, target_fake)
            
            # factor of 0.5 like in paper
            loss_D_A = (loss_real_A + loss_fake_A) * 0.5 
            self.log('train/loss_D_A', loss_D_A, prog_bar=True)
            loss_D_B = (loss_real_B + loss_fake_B) * 0.5
            self.log('train/loss_D_B', loss_D_B, prog_bar=True)
            
            # original takes backward over losses seperatly; is this equivalent?
            joint_D_loss = loss_D_A + loss_D_B
            
            return joint_D_loss

    def validation_step(self, batch, batch_idx):
        '''computation of the validation data'''
        lambda_A = self.hparams.cyc_lambda_A
        lambda_B = self.hparams.cyc_lambda_B
        lambda_id = self.hparams.lambda_id
        real_label = self.hparams.real_label
        fake_label = self.hparams.fake_label
        
        A, B = batch['A'], batch['B']
        bs = A.shape[0]
        
        if self.val_A == None and self.val_B == None:
            self.val_A = A.clone()
            self.val_B = B.clone()
        
        # log
        self._log_sequence('val/true_A', A[0].squeeze(), 0)
        self._log_sequence('val/true_B', B[0].squeeze(), 0)

        fake_B, fake_A, rec_A, rec_B = self.gen_all_sequences(A, B, 'val')
        
        pred_fake_B = self.D_B(fake_B)
        target = th.tensor(real_label).expand_as(pred_fake_B).type_as(A)

        loss_G_AB = self.adverserial_ls_loss(pred_fake_B, target)
        self.log('val/G_AB', loss_G_AB)

        pred_fake_A = self.D_A(fake_A)
        target = th.tensor(real_label).expand_as(pred_fake_A).type_as(B)

        loss_G_BA = self.adverserial_ls_loss(pred_fake_A, target)
        self.log('val/G_BA', loss_G_BA)

        loss_cyc_A = self.cycle_consistency_loss(rec_A, A)
        self.log('val/Cycle_loss_A', loss_cyc_A)

        
        loss_cyc_B = self.cycle_consistency_loss(rec_B, B)
        self.log('val/Cycle_loss_B', loss_cyc_B)

        #trans_B, trans_A = self(B, A)
        #loss_id = self.cycle_consistency_loss(trans_B, B) * lambda_id + self.cycle_consistency_loss(trans_A, A) * lambda_id
        #self.log('val/identity_loss', loss_id)

        loss_generator = loss_G_AB + loss_G_BA + loss_cyc_A * lambda_A + loss_cyc_B * lambda_B# + loss_id
        self.log('val/generator_loss', loss_generator)
        self.log('hp_metric', loss_generator)


        target_real = th.full((bs, 1), real_label).type_as(A)

        pred_real_A = self.D_A(A)
        loss_real_A = self.adverserial_ls_loss(pred_real_A, target_real)
        
        target_real = th.full((bs, 1), real_label).type_as(B)
        pred_real_B = self.D_B(B)
        loss_real_B = self.adverserial_ls_loss(pred_real_B, target_real)

        pred_fake_A = self.D_A(fake_A)
        target_fake = th.tensor(fake_label).expand_as(pred_fake_A).type_as(A)
        loss_fake_A = self.adverserial_ls_loss(pred_fake_A, target_fake)

        pred_fake_B = self.D_B(fake_B)
        target_fake = th.tensor(fake_label).expand_as(pred_fake_B).type_as(B)
        loss_fake_B = self.adverserial_ls_loss(pred_fake_B, target_fake)

        loss_D_A = (loss_real_A + loss_fake_A) * 0.5 
        self.log('val/loss_D_A', loss_D_A)
        loss_D_B = (loss_real_B + loss_fake_B) * 0.5
        self.log('val/loss_D_B', loss_D_B)

        self._log_spectra(A[0], B[0], fake_A[0], fake_B[0], rec_A[0], rec_B[0])
        
    def on_validation_epoch_end(self) -> None:
        '''Generates validation images tracked over whole training'''
        if self.val_A != None and self.val_B != None and (self.current_epoch % 10) == 0:
            A = self.val_A.type_as(self.G_AB.ex_weights)
            B = self.val_B.type_as(self.G_BA.ex_weights)
            
            self.gen_all_sequences(A, B, 'gen', self.current_epoch)


    def gen_all_sequences(self, A, B, stage='train', idx=0):
            ''' generates all images of the CycleGAN and logs them to the fitting stage
            Args:
                A (th.tensor): input images from domain A 
                B (th.tensor): input images from domain B 
            '''
            fake_B, fake_A = self(A=A, B=B)
            # log
            self._log_sequence(f"{stage}/fake_B", fake_B[0].squeeze(), idx)
            self._log_sequence(f"{stage}/fake_A", fake_A[0].squeeze(), idx)
            
            # add noise before reconstruction
            noisy_fake_B = fake_B +  th.rand_like(fake_B) * 0.1
            noisy_fake_A = fake_A + th.rand_like(fake_A) * 0.1

            rec_B, rec_A = self(A=fake_A, B=fake_B)
            # log
            self._log_sequence(f"{stage}/rec_B", rec_B[0].squeeze(), idx)
            self._log_sequence(f"{stage}/rec_A", rec_A[0].squeeze(), idx)

            
            return fake_B, fake_A, rec_A, rec_B

    def _log_sequence(self, label, sequence, idx):
            fig, ax = plt.subplots()
            ax.plot(sequence.cpu().detach())
            self.logger.experiment.add_figure(label, fig, idx)

    def _log_spectra(self, A, B, fake_A, fake_B, rec_A, rec_B, idx=0):
        self._log_spectrum(A[0], 'FFT/A_true')
        self._log_spectrum(fake_B[0], 'FFT/B_fake')
        self._log_spectrum(rec_A[0], 'FFT/A_rec')

        self._log_spectrum(B[0], 'FFT/B_true')
        self._log_spectrum(fake_A[0], 'FFT/A_fake')
        self._log_spectrum(rec_B[0], 'FFT/B_rec')
        
    def _log_spectrum(self, X, title, idx=0):    
        # fft models positive frequencies up to the middle entry
        # here also only frequencies up to 65 are present
        selected_frequencies = 67

        fft = th.fft.rfft(X.squeeze()).abs()[:selected_frequencies].cpu()
        fig, ax = plt.subplots()
        ax.plot(fft)
        self.logger.experiment.add_figure(title, fig, idx)
        
    @staticmethod
    def update_parseargs(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('-x', '--decay-epochs', default=100, type=int, help='total number of epochs to train using linear decay')       ## NOTE: this is currently disabled
        parser.add_argument('-s', '--smoothing', default=1, type=float, help='number to use as the smoothed label. (default: 1 - no smoothing)')
        parser.add_argument('-D', '--lr-d', default=0.0002, type=float, help='Learning rate of the discriminator')
        parser.add_argument('-G', '--lr-g', default=0.0002, type=float, help='Learning rate of the generator')
        parser.add_argument('-y', '--layers', default=2, type=int, help='Depth of the Generator')
        parser.add_argument('-r', '--residuals', default=3, type=int, help='number of residual blocks')
        parser.add_argument('--dilations', action='store_true', help='activates dilations and attention blocks')
        parser.add_argument('--fileA', type=str, help='Path to ds A')
        parser.add_argument('--fileB', type=str, help='Path to ds B')
        parser.add_argument('--mode', '-m', type=str, default='unet', help='sets the operation mode of the GAN generator. one of: "unet", "resnet"')

        return parser, TCArgs

class TCArgs(Arguments):
    decay_epochs: int
    smoothing: float
    lr_d: float
    lr_g: float
    residuals: int
    dilations: bool
    fileA: str
    fileB: str
    layers: int
    mode: str

    def __str__(self):
        s = super(TCArgs, self).__str__()
        arr = s.split('\n')
        arr += [
            f'learning rates G: {self.lr_g} D: {self.lr_d}',
            f'conv depth {self.layers}',
            f'latent residuals {self.residuals}',
            f'generator mode {self.mode}' + '' if not self.dilations else ' with dilations and attention'
        ]

        return '\n'.join(arr)


