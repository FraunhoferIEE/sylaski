import numpy as np
import torch
from torch import nn


class MomentLoss(nn.Module):
    """
    Criterion that measures the discrepancy between mean and standard deviation of input and target.
    Args:
        axis: integer defining over which axis the moments shall be calculated. Default is 2.
    """

    def __init__(self, axis: int = 2):
        super().__init__()
        self.axis = axis

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Computes the discrepancy between the means as well as the variances of the two input tensors.
        Args:
            input: torch.tensor. first input tensor
            target: torch.tensor. second input tensor
        Returns:
            moment_loss: sum of mean and variance differences

        """
        # loss of standard deviation discrepancies
        std_loss = torch.abs(input.std(axis=self.axis) - target.std(axis=self.axis)).mean()
        # loss of mean discrepancies
        mean_loss = torch.abs(input.mean(axis=self.axis) - target.mean(axis=self.axis)).mean()
        moment_loss = std_loss + mean_loss
        return moment_loss


class WGANGPLoss(nn.Module):
    """
    WGANGP Loss. Needs as input the critic (dis_model) as well as a batch of real and generated samples.
    Args:
        lambda_gp: penalty parameter in gradient penalty term of critic loss
    """

    def __init__(self, lambda_gp: float):
        super().__init__()
        self.lambda_gp = lambda_gp

    def compute_gp(self, x_real, x_gen, dis_model, *args):
        """
        Computation of the gradient penalty term in the loss of the discriminator.
        """
        alpha = np.random.random()  # random number in [0,1]
        interpolate = alpha * x_real + (1 - alpha) * x_gen  # interpolation between real and generated data
        interpolate.requires_grad_(True)
        # discriminator's output on interpolation
        d_inter = dis_model(interpolate, *args)
        grad_outputs = torch.ones(x_real.size(0), 1).requires_grad_(False).float().to(d_inter)
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

    def forward(self, dis_model, x_real: torch.Tensor, x_gen: torch.Tensor, *args,
                step: int = 1):
        """
        Forward pass in the loss computation.
        Args:
             dis_model: discriminator model
             x_real: batch of real samples
             x_gen: batch of generated samples
             *args: additional arguments given as input to the models (e.g. labels).
             step: whether to calculate loss for generator (0) or discriminator (1).
                Options: {0, 1} or {'discriminator', 'generator'}
        """
        x_real = x_real.to(x_gen)
        if step == 1 or step == 'discriminator':
            gp = self.compute_gp(x_real, x_gen.detach(), dis_model, *args)
            d_loss = torch.mean(dis_model(x_gen.detach(), *args)) - torch.mean(dis_model(x_real, *args)) \
                     + self.lambda_gp * gp
            return d_loss
        elif step == 0 or step == 'generator':
            g_loss = - torch.mean(dis_model(x_gen, *args))
            return g_loss


class LSLoss(nn.Module):
    """
    Least-Squares Loss.
    """

    def __init__(self, valid_label: int = 1, fake_label: int = 0):
        super().__init__()
        self.valid_label = valid_label
        self.fake_label = fake_label

    def forward(self, dis_real, dis_gen, step: str = 'discriminator'):
        """
        Forward pass of Loss.
        Args:
            dis_real: output of discriminator on real samples
            dis_gen: output of discriminator on fake samples
            step: Options: {'discriminator', 'generator'}
        """
        if step == 'discriminator':
            d_loss = 0.5 * (
                    torch.mean((dis_real - self.valid_label) ** 2) + torch.mean((dis_gen - self.fake_label) ** 2))
            return d_loss
        elif step == 'generator':
            g_loss = torch.mean((dis_gen - self.valid_label) ** 2)
            return g_loss
