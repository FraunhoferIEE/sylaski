import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from torch.utils.data import DataLoader


def create_dataloader(dataset, step: int = 0, shuffle: bool = True, batch_size: int = 32):
    window_length = int(2 ** (step + 3))
    dataset.window_length = window_length
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    return loader


def load_real_data(dataset, size: int = 1000):
    """
    Randomly loads a number of *size* samples from the given dataset.
    """

    idx = np.random.permutation(len(dataset))[:size]
    data = dataset.data
    return data[idx]


def plot_losses(history: dict, save: bool = False, save_path: str = '/image/losses/'):
    """
    Plots losses of critic/discriminator and generator for each training epoch.
    Args:
        history: dictionary containing the losses of discriminator and generator
        save: boolean whether to save the plot in the given directory
        save_path: directory where plot is saved, if needed
    """
    critic_loss = history['d_loss']
    gen_loss = history['g_loss']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
    ax1.semilogx(critic_loss, color='black', linewidth=0.5)
    ax1.set_title('Critic Loss')
    ax2.semilogx(gen_loss, color='black', linewidth=0.5)
    ax2.set_title('Generator Loss')
    ax2.set_xlabel('epochs')
    sns.despine(left=True, bottom=True, right=True)

    if save:
        os.makedirs(save_path, exist_ok=True)
        file_name = 'History_{}.png'.format(name)
        plt.savefig(save_path + file_name)
    fig.tight_layout()
    return fig


def find_most_similar_sequence(sample_real: torch.Tensor, data_fake: torch.Tensor, criterion: str = 'mse'):
    """
    Finds the 'most similar' sample for a number of generated sequences, given a real sequence.
    Args:
        sample_real: tensor of shape [n_features, seq_len] of a real sample
        data_fake: tensor of shape [n_samples, n_features, seq_len] containing a number of fake samples to be compared
                    to the real sample
        criterion: options: 'mse' TODO: add other options here
    """
    if criterion == 'mse':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError
    distances = torch.zeros(len(data_fake))
    for i in range(len(data_fake)):
        loss = criterion(sample_real, data_fake[i])
        distances[i] = loss

    idx = torch.argmin(distances)

    return data_fake[idx]


def plot_similar_samples(data_real: torch.Tensor, data_fake: torch.Tensor, n_rows: int = 4, n_cols: int = 4,
                         criterion: str = 'mse', figsize=(8, 25)):
    """
    Randomly uses a sample from the real data to find the 'most similar' generated sample using a given criterion.
    Plots both samples with all features.
    Args:
        data_real: torch.Tensorcontaining real data
        data_fake: torch.Tensor containing generated samples
        feature_names: List of feature names
        criterion: criterion by which similarity between individual samples is measured.
    Returns:
        fig: matplotlib.pyplot.figure

    """
    idx = np.random.permutation(len(data_real))[:n_rows * n_cols]
    x_real = data_real[idx].squeeze()

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].plot(x_real[count], color='black', label='Real')
            x_fake = find_most_similar_sequence(x_real[count], data_fake.squeeze(), criterion=criterion)
            axs[i, j].plot(x_fake, color='r', label='Fake')
            count += 1
    plt.legend()
    fig.text(0.5, 0.07, 'time steps', ha='center', va='center', fontsize=17)
    fig.suptitle("Real and generated feature series", fontsize=19, y=0.95)

    return fig
