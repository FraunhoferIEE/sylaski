import torch
import numpy as np
from collections import defaultdict

from syndatagenerators.models.wgan.wgan_model import WGAN

class WGANGP(WGAN):
    """
    GAN with gradient penalty. Instead of weight clipping, it uses  a gradient penalty term to enforce the Lipschitz
    condition on the Critic.

    """
    def __init__(self, params: dict):
        super(WGANGP, self).__init__(
            params)
        self.lambda_gp = params['train_params']['lambda_gp']

    def compute_gp(self, x_real, x_gen):
        """
        Computes the gradient penalty term in the loss of the Critic.
        """
        alpha = np.random.random()  # random number in [0,1]
        interpolate = alpha * x_real + (1 - alpha) * x_gen  # interpolation between real and generated data
        interpolate.requires_grad_(True)
        d_inter = self.discriminator(interpolate)  # discriminator's output on interpolation
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
        gradients_penalty = (((gradients+1e-6).norm(2, dim=1) - 1) ** 2).mean()  # add small epsilon term for stability
        return gradients_penalty

    def train(self, loader):
        if not self.generator:
            raise ValueError("model doesn't be initialized,\
                             please call build_model() before train()")
        if self.history is None:
            self.history = defaultdict(list)
        train_params = self.train_params
        epochs = train_params['epochs']
        sample_cycle = train_params['sample_cycle']
        target: int = train_params['name']
        print("Start Training GAN with gradient penalty...")
        for epoch in range(epochs):
            local_history = self.train_on_epoch(loader)
            self.update_history(local_history)
            if (epoch + 1) % sample_cycle == 0 or (epoch + 1) == epochs:
                self.save_checkpoint(name=target)
                self.print_local_history(
                    epoch + 1, local_history, max_epoch=epochs)
        self.save_checkpoint(name=target)
        print("Done! :)")

    def train_on_epoch(self, loader):
        local_history = dict()
        tmp_history = defaultdict(list)
        for i, x_batch in enumerate(loader):
            x_batch = x_batch.to(self.device).float()
            batch_size = x_batch.size(0)
            z = self.gen_noise(0, 1, (batch_size, self.latent_dim)).float()
            x_gen = self.generator(z)
            self.optimizer_d.zero_grad()
            gradients_penalty = self.compute_gp(x_batch, x_gen)
            d_loss = torch.mean(self.discriminator(x_gen)) - torch.mean(self.discriminator(x_batch)) \
                     + self.lambda_gp * gradients_penalty  # loss of critic: add gradient penalty term
            d_loss.backward(retain_graph=True)
            self.optimizer_d.step()
            tmp_history['d_loss'].append(d_loss.item())
            if i % self.n_critic == 0:
                self.optimizer_g.zero_grad()
                g_loss = - torch.mean(self.discriminator(x_gen))
                g_loss.backward()
                self.optimizer_g.step()
                tmp_history['g_loss'].append(g_loss.item())

        local_history['d_loss'] = np.mean(tmp_history['d_loss'])
        local_history['g_loss'] = np.mean(tmp_history['g_loss'])
        return local_history