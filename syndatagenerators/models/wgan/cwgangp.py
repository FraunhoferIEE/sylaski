import torch
import os
import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm
from collections import defaultdict


class CCritic(nn.Module):
    """ Conditional Critic: Replaces the discriminator network in the GAN. Additionally, it is conditioned
        on a label which is fed into the network together with the input sample in a joint representation.

    """

    def __init__(self, params: dict, class_size: int = 2):
        """
        Input:
        params: dictionary of critic parameters.
        embedding_dim: dimension of embedding of labels
        class_size: number of (different) classes (labels)
        """
        super(CCritic, self).__init__()
        self.input_shape = params['input_shape']
        self.features = self.input_shape[0]  # features of the time series
        self.relu_slope = 0.2  # slope of leaky ReLU in sequential block
        self.drop_rate = params['drop_rate']
        self.stride = params['stride']
        self.kernel_size = params['kernel_size']
        self.padding = params['padding']
        self.dilation = params['dilation']
        self.class_size = class_size
        self.embedding_dim = self.input_shape[1]

        def dis_block(in_channel, out_channel):
            """


            """
            layers = [
                spectral_norm(
                    nn.Conv1d(in_channel, out_channel, self.kernel_size, stride=self.stride, padding=self.padding,
                              dilation=self.dilation), n_power_iterations=10),
                nn.LeakyReLU(self.relu_slope),
            ]

            return layers

        self.embedding = nn.Embedding(self.class_size, self.embedding_dim)  # embedding function of label

        self.model = nn.Sequential(
            *dis_block(self.features + 1, 16),  # change dimension for joint representation: add channel for labels
            *dis_block(16, 32),
            *dis_block(32, 64),
            *dis_block(64, 128),
        )
        ds_size = int(np.ceil(self.input_shape[1] / 2 ** 4))
        self.fc = nn.Sequential(
            nn.Linear(128 * ds_size, 64),
            nn.LeakyReLU(self.relu_slope),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 1),
        )

    def forward(self, x, label):
        """
        Args:
            x: batch of samples of shape [batch_size, feature_dim, seq_len]
            label: class label. Here, label is in {0,1}.

        """
        label = self.embedding(label)
        if x.dim() == 2:
            x = x.view(x.size(0), 1, -1)
        x_joint = torch.cat([x, label], 1)  # concatenate input sample and label in joint hidden representation
        out = self.model(x_joint)
        out = out.view(out.size(0), -1)
        feature = self.fc(out)
        validity = self.fc2(feature)
        return validity


class CGenerator(nn.Module):
    """ Conditional Generator network of CWGANGP model. Takes as input a latent vector z from a normal distribution
        and a label together in a joint representation.
     """

    def __init__(self, params: dict, class_size: int = 2, relu_slope: float = 0.2):
        """
        Parameters
        ----------
        params : dictionary.
        class_size: number of different classes of the labels.
        relu_slope: negative slope of LeakyReLU activation in between layers.
        -------
        """
        super(CGenerator, self).__init__()
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
        self.embedding_dim = self.init_size
        self.class_size = class_size

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 128 * self.init_size)
        )
        self.embedding = nn.Embedding(self.class_size, self.init_size)

        if self.normalization == 'batch':
            self.conv_blocks = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(128 + 1, 128, self.kernel_size, self.stride, self.padding, dilation=self.dilation),
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
                spectral_norm(
                    nn.Conv1d(128 + 1, 128, self.kernel_size, self.stride, self.padding, dilation=self.dilation),
                    n_power_iterations=10),
                nn.LeakyReLU(self.relu_slope, inplace=True),
                nn.Upsample(scale_factor=2),
                spectral_norm(nn.Conv1d(128, 64, self.kernel_size, self.stride, self.padding, dilation=self.dilation),
                              n_power_iterations=10),
                nn.LeakyReLU(self.relu_slope, inplace=True),
                spectral_norm(
                    nn.Conv1d(64, self.features, self.kernel_size, self.stride, self.padding, dilation=self.dilation),
                    n_power_iterations=10)
            )

        if self.activ_fct == 'tanh':
            self.activation = nn.Tanh()
        elif self.activ_fct == 'relu':
            self.activation = nn.ReLU()
        elif self.activ_fct == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activ_fct == 'linear:':
            self.activation = nn.Linear(self.features, self.features)

        else:
            self.activation = None

    def forward(self, z, label):
        out = self.fc(z)  # linear layer
        out = out.view(out.size(0), 128, self.init_size)  # reshape to 3 dim
        label = self.embedding(label)
        out_joint = torch.cat([out, label], 1)  # concatenate z and label
        x = self.conv_blocks(out_joint)  # conv layers
        if self.activation is not None:  # activation
            x = self.activation(x)
        return x


class CWGANGP(object):
    """ class of conditional WGANGP model. """

    def __init__(self, params):

        super().__init__()
        train_params = params['train_params']  # training parameters
        dis_params = params['dis_params']  # critic parameters
        gen_params = params['gen_params']  # generator parameters
        self.train_params = train_params
        self.dis_params = dis_params
        self.gen_params = gen_params
        self.noise_type = gen_params['noise_type']  # distribution of latent noise
        self.lambda_gp = train_params['lambda_gp']  # gradient penalty parameter
        self.input_shape = train_params['input_shape']
        self.latent_dim = gen_params['latent_dim']
        self.lr = train_params['lr']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = train_params['name']  # integer name under which model is saved
        self.optimizer = train_params['optimizer']
        self.opt_args = train_params['opt_args']
        self.activation = gen_params['activ_fct']  # activation function for generator
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

    def compute_gp(self, x_real, x_gen, labels):
        """
        Computes the gradient penalty term in the loss of the Critic.
        """
        alpha = np.random.random()  # random number in [0,1]
        interpolate = alpha * x_real + (1 - alpha) * x_gen  # interpolation between real and generated data
        interpolate.requires_grad_(True)
        d_inter = self.discriminator(interpolate, labels)  # discriminator's output on interpolation + labels
        grad_outputs = self.gen_tensor(np.ones((x_real.size(0), 1)))
        gradients = torch.autograd.grad(
            outputs=d_inter,
            inputs=interpolate,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]  # computation of gradients
        gradients = gradients.view(gradients.size(0), -1)
        gradients_penalty = (
                    ((gradients + 1e-6).norm(2, dim=1) - 1) ** 2).mean()  # add small epsilon term for stability
        return gradients_penalty

    def train(self, loader):
        """
        trains model with specified dataloader.

        Parameters
        ----------
        loader : of type torch.utils.data.DataLoader
            Dataloader iterating over train data.
        """
        if not self.generator:
            raise ValueError("model doesn't be initialized,\
                             please call build_model() before train()")
        if self.history is None:
            self.history = defaultdict(list)
        train_params = self.train_params
        epochs = train_params['epochs']
        sample_cycle = train_params['sample_cycle']
        target: int = train_params['name']
        [m.train() for m in self.models.values()]

        print("Start Training...")
        for epoch in range(epochs):
            local_history = self.train_on_epoch(loader)
            self.update_history(local_history)
            if (epoch + 1) % sample_cycle == 0 or (epoch + 1) == epochs:
                self.save_checkpoint(name=target)
                self.print_local_history(
                    epoch + 1, local_history, max_epoch=epochs)
        self.save_checkpoint(name=target)
        print("Done! :)")
        return

    def train_on_epoch(self, loader):
        local_history = dict()
        tmp_history = defaultdict(list)
        for i, (x_batch, labels) in enumerate(loader):  # dataloader must contain batches and labels
            batch_size = x_batch.size(0)
            x_batch = x_batch.to(self.device).float()
            # fake_labels = torch.randint(0, 1, (batch_size,)).to(self.device) # use fake labels for generator?
            z = self.gen_noise(0, 1, (batch_size, self.latent_dim)).float()
            x_gen = self.generator(z, labels)  # batch output of generator
            self.optimizer_d.zero_grad()  # optimizer of discriminator: set gradients to zero
            # GAN loss: - mean(D(real)) + mean(D(fake))
            # discriminator loss: compute gradient penalty
            gradients_penalty = self.compute_gp(x_batch, x_gen, labels)
            d_loss = torch.mean(self.discriminator(x_gen, labels)) - torch.mean(self.discriminator(x_batch, labels)) \
                     + self.lambda_gp * gradients_penalty
            d_loss.backward(retain_graph=True)
            self.optimizer_d.step()
            tmp_history['d_loss'].append(d_loss.item())
            if i % self.n_critic == 0:  # train generator every (n_critic)th time
                self.optimizer_g.zero_grad()  # set gradients to zero
                g_loss = - torch.mean(self.discriminator(x_gen, labels))
                g_loss.backward()
                self.optimizer_g.step()
                tmp_history['g_loss'].append(g_loss.item())

        local_history['d_loss'] = np.mean(tmp_history['d_loss'])
        local_history['g_loss'] = np.mean(tmp_history['g_loss'])
        return local_history

    def sample(self, label, clip: bool = True):
        """

        Returns a number of generated samples.
        Args:
            label: class label of generated samples.
            size: number of generated samples.
            clip: If generated samples shall be clipped to the given interval.

        """
        size = len(label)
        train_params = self.train_params
        ivl = train_params['ivl']
        [m.eval() for m in self.models.values()]
        noise = self.gen_noise(0, 1, (size, self.latent_dim))
        x_fake = self.generator(noise, label).detach()  # sample from generator
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
