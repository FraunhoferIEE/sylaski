from math import log2

import torch
import numpy as np
from pytorch_lightning import Callback
import torch.nn.functional as F

from syndatagenerators.metrics.discriminative_score import discriminative_score
from syndatagenerators.metrics.mmd_score import mmd
from syndatagenerators.models.utils.plot_functions import plot_acor_per_feature
from syndatagenerators.models.progressive_gan import TrainerProGAN
from syndatagenerators.models.cond_progressive_gan import TrainerCPGAN


class DiscriminativeCallback(Callback):
    """
    Callback that can be used in trainer in order to evaluate the generated samples using the discriminative score.

    """

    def __init__(self, size: int = 300):
        super().__init__()
        self.size = size

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx: int, dataloader_idx: int) -> None:

        x_real, label = process_batch(batch, pl_module, self.size)

        x_gen = generate_samples(pl_module, self.size, label)

        if isinstance(pl_module, (TrainerProGAN, TrainerCPGAN)):
            reduce_factor = int(log2(pl_module.target_len)) - int(log2(x_gen.size(2)))
            x_real = F.avg_pool1d(x_real, kernel_size=2 ** reduce_factor)

        dis_score = discriminative_score(x_real, x_gen, n_epochs=30, layer_dim=2, enhanced_output=False, patience=20)
        pl_module.log("discriminative_loss", dis_score, on_epoch=True)


class MMDCallback(Callback):
    """
    Callback evaluating the Maximum-Mean-Discrepancy between a batch of real samples and a batch of generated samples.
    Args:
        size: integer defining number of samples to compare.
    """

    def __init__(self, size: int = 1000):
        super().__init__()
        self.size = size

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx: int, dataloader_idx: int) -> None:

        x_real, label = process_batch(batch, pl_module, self.size)

        x_gen = generate_samples(pl_module, self.size, label)

        if isinstance(pl_module, (TrainerProGAN, TrainerCPGAN)):
            reduce_factor = int(log2(pl_module.target_len)) - int(log2(x_gen.size(2)))
            x_real = F.avg_pool1d(x_real, kernel_size=2 ** reduce_factor)

        mmd_score = mmd(x_real, x_gen)
        pl_module.log('mmd', mmd_score, on_epoch=True)


class ACFCallback(Callback):
    """
    Callback evaluating the Autocorrelation of real and synthetic data.
    Args:
        size: number of samples for which the (average) autocorrelation is calculated for
    """

    def __init__(self, size: int = 1000):
        super().__init__()
        self.size = size

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx: int, dataloader_idx: int) -> None:

        x_real, label = process_batch(batch, pl_module, self.size)

        x_gen = generate_samples(pl_module, self.size, label)

        if isinstance(pl_module, (TrainerProGAN, TrainerCPGAN)):
            reduce_factor = int(log2(pl_module.target_len)) - int(log2(x_gen.size(2)))
            x_real = F.avg_pool1d(x_real, kernel_size=2 ** reduce_factor)

        lag = x_real.size(2) - 1

        fig_corr = plot_acor_per_feature(x_real, x_gen, max_lag=lag)

        pl_module.logger.experiment.add_figure(f'Autocorrelation real vs generated data', fig_corr,
                                               pl_module.current_epoch)


def process_batch(batch, pl_module, size: int = 1000):
    """
    processes the batch of the dataloader.
    """
    # set label to None in the beginning
    label = None

    if len(batch) == 2:
        x_batch, label = batch
    else:
        x_batch = batch

    idx = np.random.randint(0, x_batch.size(0), size)
    x_batch = x_batch.to(pl_module.device)[idx]
    label = label.to(pl_module.device)[idx] if label is not None else None

    return x_batch, label


def generate_samples(pl_module, size: int, label=None)  :
    """
    generates synthetic samples given the respective lightning module, size and optionally labels.
    Args:
        pl_module: pytorch lightning module used for training. Can be of class {TrainerProGAN, TrainerCPGAN }
        size: number of samples to generate
        label: optional label to be given into the model
    """
    if isinstance(pl_module, (TrainerProGAN, TrainerCPGAN)):
        depth = pl_module.depth
        residual = pl_module._residual(pl_module.current_epoch)
        z = torch.randn(size, pl_module.feature_dim, pl_module.target_len).to(pl_module.device)
        if label is not None:
            x_gen = pl_module(z, label, depth, residual)
        else:
            x_gen = pl_module(z, depth, residual)

    else:
        raise NotImplementedError

    return x_gen



