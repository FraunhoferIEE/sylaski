import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.utils import spectral_norm
from collections import defaultdict


class Critic(nn.Module):
    """
    Critic model: Replaces the discriminator network in the GAN.
    """

    def __init__(self, params):
        """
        Input:
        params: dictionary of critic parameters.
        """
        super(Critic, self).__init__()
        self.input_shape = params['input_shape']
        self.features = self.input_shape[0]  # features of the time series
        self.relu_slope = 0.2  # slope of leaky ReLU in sequential block
        self.drop_rate = params['drop_rate']
        self.stride = params['stride']
        self.kernel_size = params['kernel_size']
        self.padding = params['padding']
        self.dilation = params['dilation']
        self.dropout = params['dropout']

        def dis_block(in_channel, out_channel):
            layers = [
                spectral_norm(
                    nn.Conv1d(in_channel, out_channel, self.kernel_size, stride=self.stride, padding=self.padding,
                              dilation=self.dilation), n_power_iterations=10),
                # use of spectral normalization helps fulfill the Lipschitz constraint
                nn.LeakyReLU(self.relu_slope),
            ]

            return layers

        self.model = nn.Sequential(
            *dis_block(self.features, 16),
            *dis_block(16, 32),
            *dis_block(32, 64),
            *dis_block(64, 128),
        )
        ds_size = int(np.ceil(self.input_shape[1] / 2 ** 4))
        # sequence length reduces in Critic per every epoch:
        # example: after 4 epochs for seq_len of 24: ds_size=ceil(24/2**4)=2
        self.fc = nn.Sequential(
            nn.Linear(128 * ds_size, 64),
            nn.LeakyReLU(self.relu_slope),
            nn.Dropout(self.drop_rate)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), 1, -1)
        out = self.model(x)
        out = out.view(out.size(0), -1)
        feature = self.fc(out)
        validity = self.fc2(feature)
        return validity


class Generator(nn.Module):
    """ Generator network of GAN model. """

    def __init__(self, params, relu_slope=0.2):
        """
        Parameters
        ----------
        params : dictionary.
        relu_slope: negative slope of LeakyReLU activation in between layers.
        -------
        """
        super(Generator, self).__init__()
        self.gen_params = params
        self.input_shape = params['input_shape']
        self.latent_dim = params['latent_dim']  # dimension of input noise
        self.activ_fct = params['activ_fct']  # used in output layer
        self.init_size = int(np.ceil(self.input_shape[1] / 4))  # for upsampling
        self.features = self.input_shape[0]  # number of features
        self.relu_slope = relu_slope  
        self.normalization = params['normalization']  # normalization: 'batch', 'spectral' or None
        self.stride = params['stride']
        self.padding = params['padding']
        self.kernel_size = params['kernel_size']
        self.dilation = params['dilation']

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 128 * self.init_size)
        )
        if self.normalization == 'batch':
            self.conv_blocks = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(128, 128, self.kernel_size, self.stride, self.padding, dilation=self.dilation),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(self.relu_slope, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(128, 64, self.kernel_size, self.stride, self.padding, dilation=self.dilation),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(self.relu_slope, inplace=True),
                nn.Conv1d(64, self.features, self.kernel_size, self.stride, self.padding,
                          dilation=self.dilation)
            )

        elif self.normalization == 'spectral':
            self.conv_blocks = nn.Sequential(
                nn.Upsample(scale_factor=2),

                spectral_norm(nn.Conv1d(128, 128, self.kernel_size, self.stride, self.padding, dilation=self.dilation),
                              n_power_iterations=10),
                nn.LeakyReLU(self.relu_slope, inplace=True),
                nn.Upsample(scale_factor=2),
                spectral_norm(nn.Conv1d(128, 64, self.kernel_size, self.stride, self.padding, dilation=self.dilation),
                              n_power_iterations=10),
                nn.LeakyReLU(self.relu_slope, inplace=True),
                nn.Conv1d(64, self.features, self.kernel_size, self.stride, self.padding, dilation=self.dilation)
            )

        if self.activ_fct == 'relu':
            self.activation = nn.ReLU()
        elif self.activ_fct == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activ_fct == 'linear':
            self.activation = nn.Linear(self.features, self.features)

        else:
            self.activation = None

    def forward(self, z):
        out = self.fc(z)  # linear layer
        out = out.view(out.size(0), 128, self.init_size)  # reshape to 3 dim
        x = self.conv_blocks(out)  # conv layers
        if self.activation is not None:  # activation
            x = self.activation(x)
        return x


class WGAN(object):
    """ class of GAN model. """

    def __init__(self, params):

        super().__init__()
        train_params = params['train_params']  # training parameters
        dis_params = params['dis_params']  # critic parameters
        gen_params = params['gen_params']  # generator parameters
        self.train_params = train_params
        self.dis_params = dis_params
        self.gen_params = gen_params
        self.noise_type = gen_params['noise_type']  # distribution of latent noise
        self.clip_val = train_params['clip_value']  # clip value for weights of critic
        self.input_shape = train_params['input_shape']
        self.latent_dim = gen_params['latent_dim']
        self.lr = train_params['lr']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = train_params['name']  # integer name under which model is saved
        self.optimizer = train_params['optimizer']
        self.opt_args = train_params['opt_args']
        self.activation = gen_params['activ_fct']
        self.normalization_dis = dis_params['normalization']
        self.normalization_gen = gen_params['normalization']
        self.n_critic = train_params['n_critic']  # n_critic: how much more often shall discriminator be trained
        self.drop_rate = dis_params['drop_rate']  # drop rate for critic

        # initialize models, optimizers, history, paths
        self.generator = None
        self.discriminator = None
        self.optimizer_g = None
        self.optimizer_d = None
        self.history = None
        self.models = dict()
        self.img_path = '../../GAN/image/'
        self.save_path = '../../GAN/ckpt/'

    def build_model(self, gen_cls, dis_cls):
        """
        initializes the model.

        Parameters
        ----------
        gen_cls :
            class of generator.
        dis_cls :
            class of discriminator.
        """

        self.generator = gen_cls(self.gen_params).to(self.device)
        self.discriminator = dis_cls(self.dis_params).to(self.device)
        self.optimizer_g = self.optimizer(self.generator.parameters(),
                                          lr=self.lr, **self.opt_args)
        self.optimizer_d = self.optimizer(self.discriminator.parameters(),
                                          lr=self.lr, **self.opt_args)
        self.models = {'generator': self.generator,
                       'discriminator': self.discriminator}

    def train(self, loader, normalize=False):
        """
        trains model with specified dataloader.

        Parameters
        ----------
        loader : of type torch.utils.data.DataLoader
            dataloader iterating over train data.
        normalize : boolean, optional
            if generated batches shall be normalized before saving. The default is False.

        """
        if not self.generator:
            raise ValueError("model not initialized,\
                             please call build_model() before train()")
        self.history = defaultdict(list)
        train_params = self.train_params
        epochs = train_params['epochs']
        sample_cycle = train_params['sample_cycle']
        target: int = train_params['name']

        print("Start Training...")
        for epoch in range(epochs):
            local_history, d_loss, g_loss = self.train_on_epoch(loader, epoch)
            self.update_history(local_history)
            if (epoch + 1) % sample_cycle == 0 or (epoch + 1) == epochs:
                self.save_checkpoint(name=target)
                self.print_local_history(
                    epoch + 1, local_history, max_epoch=epochs)
        self.save_checkpoint(name=target)
        print("Done! :)")
        return

    def train_on_epoch(self, loader, epoch):
        local_history = dict()
        tmp_history = defaultdict(list)
        for i, x_batch in enumerate(loader):
            x_batch = x_batch.to(self.device).float()
            batch_size = x_batch.size(0)
            z = self.gen_noise(0, 1, (batch_size, self.latent_dim)).float()
            x_gen = self.generator(z)  # batch output of generator
            self.optimizer_d.zero_grad()  # optimizer of discriminator: set gradients to zero
            # GAN loss: - mean(D(real)) + mean(D(fake))
            # discriminator loss
            d_loss = torch.mean(self.discriminator(x_gen.detach())) - torch.mean(self.discriminator(x_batch))
            d_loss.backward()
            self.optimizer_d.step()
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clip_val, self.clip_val)  # clip weights of discriminator
            tmp_history['d_loss'].append(d_loss.item())
            if i % self.n_critic == 0:  # train generator every (n_critic)th time
                self.optimizer_g.zero_grad()  # set gradients to zero
                g_loss = - torch.mean(self.discriminator(x_gen))
                g_loss.backward()
                self.optimizer_g.step()
                tmp_history['g_loss'].append(g_loss.item())

        local_history['d_loss'] = np.mean(tmp_history['d_loss'])
        local_history['g_loss'] = np.mean(tmp_history['g_loss'])
        return local_history, d_loss.item(), g_loss.item()

    def sample(self, size=1, clip=True):
        """
        return a number of generated samples.

        """
        train_params = self.train_params
        ivl = train_params['ivl']
        noise = self.gen_noise(0, 1, (size, self.latent_dim))
        x_fake = self.generator(noise).detach()  # sample from generator
        if clip:  # if clip == True, values of generated samples are clipped at the interval boundaries
            x_fake = torch.clamp(x_fake, ivl[0], ivl[1])
        return x_fake

    def save_checkpoint(self, name=None):
        if name is None:
            name = '{:s}.pkl'.format(self.__class__.__name__)
        else:
            name = '{:s}_{:d}.pkl'.format(self.__class__.__name__, name)
        os.makedirs(self.save_path, exist_ok=True)
        model_state = dict()
        for k, v in self.models.items():
            model_state[k] = v.state_dict()
        model_state['history'] = self.history
        torch.save(model_state, self.save_path + name)

    def update_history(self, local_history):
        for k, v in local_history.items():
            self.history[k].append(v)

    def plot_history(self, save=True):
        os.makedirs(self.img_path, exist_ok=True)
        r = len(self.history.keys())
        fig = plt.figure(figsize=(15, int(r * 3)))
        for i, k in enumerate(self.history.keys()):
            plt.subplot(r, 1, i + 1)
            plt.semilogx(self.history[k])
            plt.title(k)
        if save:
            if self.name is None:
                plt.savefig(self.img_path +
                            'history_{:s}.png'.format(self.__class__.__name__))
            else:
                plt.savefig(
                    self.img_path +
                    'history_{:s}_{:d}.png'.format(self.__class__.__name__, self.name))
                plt.close()
        return fig

    @staticmethod
    def print_local_history(epoch, local_history, max_epoch=1000):
        num = len(str(max_epoch))
        s = 'Epoch-{:0>{}d}:  '.format(epoch, num)
        for k, v in local_history.items():
            s = s + '{}={:.4f}  '.format(k, np.mean(v))
        print(s)

    def gen_tensor(self, x, astype='float', requires_grad=False):
        if isinstance(x, torch.Tensor):
            t = x.clone().requires_grad_(requires_grad)
        else:
            t = torch.tensor(x, requires_grad=requires_grad)
        if astype == 'float':
            t = t.float()
        elif astype == 'long':
            t = t.long()
        else:
            raise ValueError('input correct astype')
        return t.to(self.device)

    def gen_noise(self, *args, **kws):
        tmp = None
        if self.noise_type == 'normal':
            tmp = np.random.normal(*args, **kws)
        elif self.noise_type == 'uniform':
            tmp = np.random.uniform(*args, **kws)
        elif self.noise_type == 'lognormal':
            tmp = np.random.lognormal(*args, *kws)
        return self.gen_tensor(tmp)

    def load_model(self):
        name = self.name
        if not self.generator:
            raise NameError("model doesn't be initialized")
        if name is None:
            path = self.save_path + '{:s}.pkl'.format(self.__class__.__name__)
        else:
            path = self.save_path + '{:s}_{:d}.pkl'.format(self.__class__.__name__, name)
        states = torch.load(path, map_location=self.device)
        for k, v in self.models.items():
            v.load_state_dict(states[k])
        self.history = states['history']
