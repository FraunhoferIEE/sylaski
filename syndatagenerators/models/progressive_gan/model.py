import os

from collections import defaultdict

import numpy as np
import torch

from syndatagenerators.models.progressive_gan.networks import ProCritic, ProGenerator


class ProGANGP:
    """
    Class of the Progressive GAN.
    """

    def __init__(self, target_len: int, nb_features: int, kernel_dis: int, kernel_gen: int, channels_dis: int,
                 channels_gen: int, residual_factor: float = 0.0, n_critic: int = 5, lambda_gp: float = 10.0,
                 lr: float = 0.0001, optimizer=torch.optim.Adam, opt_args={'betas': (0.5, 0.999)}):
        """
        Args:
            target_len: target length of the generated sequences.
            nb_features: number of features of the samples.
            kernel_dis: kernel length of the discriminator.
            kernel_gen: kernel length of the generator.
            channels_dis: hidden channel dimension in the conv blocks of the discriminator.
            channels_gen: hidden channel dimension in the conv blocks of the generator.
            residual_factor: residual factor.
            n_critic: how man times the critic shall be trained more than the generator.
            lambda_gp: gradient penalty in the loss function.
            lr: learning rate of the two models.

        """
        self.target_len = target_len
        self.nb_features = nb_features
        self.kernel_dis = kernel_dis
        self.kernel_gen = kernel_gen
        self.channels_dis = channels_dis
        self.channels_gen = channels_gen
        self.residual_factor = residual_factor
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lr = lr
        self.optimizer = optimizer
        self.opt_args = opt_args
        self.generator = None
        self.discriminator = None
        self.optimizer_g = None
        self.optimizer_d = None
        self.history = None
        self.models = dict()

    def build_model(self, gen_cls, dis_cls):
        """
        Instantiates the class with the two neural networks.
        Args:
            gen_cls : class of the generator
            dis_cls: class of the discriminator/critic.
        """
        self.generator = gen_cls(target_len=self.target_len, nb_features=self.feature_dim,
                                 channel_nb=self.channel_dim_gen, kernel_size=self.kernel_size_gen,
                                 residual_factor=self.residual_factor)

        self.discriminator = dis_cls(target_len=self.target_len, nb_features=self.feature_dim,
                                     channel_nb=self.channel_dim_dis, kernel_size=self.kernel_size_dis,
                                     residual_factor=self.residual_factor)
        self.optimizer_g = self.optimizer(self.generator.parameters(),
                                          lr=self.lr, **self.opt_args)
        self.optimizer_d = self.optimizer(self.discriminator.parameters(),
                                          lr=self.lr, **self.opt_args)
        self.models = {'generator': self.generator,
                       'discriminator': self.discriminator}

    def compute_gp(self, x_real, x_gen, step: int = 0, residual: bool = False):
        alpha = np.random.random()
        interpolate = alpha * x_real + (1 - alpha) * x_gen
        interpolate.requires_grad_(True)
        d_inter = self.discriminator(interpolate, depth=step, residual=residual)
        grad_outputs = torch.ones((x_real.size(0), 1), dtype=float).requires_grad_(False).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_inter,
            inputs=interpolate,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]  # computation of gradients
        gradients = gradients.view(gradients.size(0), -1)
        gp = (((gradients + 1e-6).norm(2, dim=1) - 1) ** 2).mean()  # add small epsilon term for stability
        return gp

    def train_on_epoch(self, loader, step: int = 0, residual: bool = False):
        [m.train() for m in self.models.values()]
        local_history = dict()
        tmp_history = defaultdict(list)
        for i, x_batch in enumerate(loader):
            x_batch = x_batch.to(self.device).float()
            batch_size = x_batch.size(0)
            z = torch.randn(batch_size, self.nb_features, self.target_len).float().to(self.device)
            x_gen = self.generator(z, depth=step, residual=residual)
            # train critic
            self.optimizer_d.zero_grad()
            gp = self.compute_gp(x_batch, x_gen, step, residual)
            d_loss = torch.mean(self.discriminator(x_gen.detach(), depth=step, residual=residual)) \
                     - torch.mean(self.discriminator(x_batch, depth=step, residual=residual)) \
                     + self.lambda_gp * gp
            d_loss.backward(retain_graph=True)
            self.optimizer_d.step()
            tmp_history['d_loss'].append(d_loss.item())
            # train generator every i-th epoch
            if i % self.n_critic == 0:
                self.optimizer_g.zero_grad()
                g_loss = - torch.mean(self.discriminator(x_gen, depth=step, residual=residual))
                g_loss.backward()
                self.optimizer_g.step()
                tmp_history['g_loss'].append(g_loss.item())

        local_history['d_loss'] = np.mean(tmp_history['d_loss'])
        local_history['g_loss'] = np.mean(tmp_history['g_loss'])

        return local_history

    def load_model(self, save_path: str):
        if not self.generator:
            raise NameError("model doesn't be initialized")
        states = torch.load(save_path, map_location=self.device)
        for k, v in self.models.items():
            v.load_state_dict(states[k])
        self.history = states['history']

    def update_history(self, local_history):
        for k, v in local_history.items():
            self.history[k].append(v)

    @staticmethod
    def print_local_history(epoch, local_history, max_epoch=1000):
        num = len(str(max_epoch))
        s = 'Epoch-{:0>{}d}:  '.format(epoch, num)
        for k, v in local_history.items():
            s = s + '{}={:.4f}  '.format(k, np.mean(v))
        print(s)

    def save_checkpoint(self, save_path: str, name=None):
        if name is None:
            name = '{:s}.pkl'.format(self.__class__.__name__)
        else:
            name = '{:s}_{:d}.pkl'.format(self.__class__.__name__, name)
        os.makedirs(save_path, exist_ok=True)
        model_state = dict()
        for k, v in self.models.items():
            model_state[k] = v.state_dict()
        model_state['history'] = self.history
        torch.save(model_state, save_path + name)

    def sample(self, size: int = 100, depth: int = 0, clip: bool = False, min=0, max=1):
        [m.eval() for m in self.models.values()]
        z = torch.randn(size, self.nb_features, self.target_len).float()
        x_gen = self.generator(z, depth=depth, residual=0).detach()
        if clip:
            x_gen = torch.clamp(x_gen, min, max)
        return x_gen

