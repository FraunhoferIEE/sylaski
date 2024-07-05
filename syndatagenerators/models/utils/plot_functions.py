import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import torch

from syndatagenerators.metrics.autocorrelation import acf


def plot_samples(x: torch.Tensor, figsize: tuple = (6, 15), title='Exemplary generated samples',
                 y_scale: str = 'uniform', labels: Optional[torch.Tensor] = None):
    """
    Returns a figure visualizing a (multivariate) tensor of generated sequences.
    Args:
        x: torch.Tensor of shape [n_batch, n_features, seq_len]
        figsize: size of figure
        title: title of plot
        y_scale: scale of y axis. If scale='uniform', it is scaled to the interval [0,1].
        labels: optional labels of the input tensor. Will be used to set a title for the individual axes.
    """
    x = x.squeeze().cpu()  # squeeze unnecessary dimensions of the tensor
    if x.dim() == 3:  # check if x has multiple features, if yes, plot only one rendomly selected sample
        # with all features
        [n_samples, n_features, seq_len] = x.shape
        idx = np.random.randint(n_samples)
        x_sample = x[idx]
        fig, axs = plt.subplots(n_features, 1, figsize=figsize, sharex=True)

        for i in range(n_features):
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
            axs[i].spines['left'].set_visible(False)
            axs[i].plot(x_sample[i])
            if y_scale == 'uniform':
                axs[i].set_ylim(0, 1)
        fig.text(0.5, 0.07, 'time steps', ha='center', va='center')
        fig.suptitle(title)

    else:
        [n_samples, seq_len] = x.shape
        fig, axs = plt.subplots(n_samples, 1, figsize=figsize, sharex=True)
        for i in range(n_samples):
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
            axs[i].spines['left'].set_visible(False)
            axs[i].plot(x[i])
            if y_scale == 'uniform':
                axs[i].set_ylim(0, 1)
            if labels is not None:
                label = labels[i].squeeze().item()
                axs[i].set_title(f'Label: {label}')

        fig.text(0.5, 0.07, 'time steps', ha='center', va='center')
        fig.suptitle(title)

        return fig

def plot_sample_grid(data, n_rows=2, n_cols=5, random_sample=True, title=None):
    """
    Plots n_total=n_rows*n_cols samples from the input data in a grid.
    Either random samples or the first n_total samples are shown.
    Args:
        data: The input data of dim 2 or 3.
        n_rows: Number of rows to plot in the grid.
        n_cols: Number of columns to plot in the grid.
        random_sample: Take a random sample from the data.
        title: The plot title.
    """
    assert len(data) >= n_rows*n_cols
    
    # Reshape and take a sample.
    if data.dim() == 3:
        data = data.view(data.shape[0], data.shape[-1])
    sample = np.random.choice([x for x in data], n_rows*n_cols) if random_sample else data[:n_rows*n_cols]
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2), sharex=True)
    count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            axs[i, j].plot(sample[count], color='black')
            count += 1
    if title is not None:
        fig.suptitle(title)
    return fig

def plot_acor_per_feature(x_real: torch.Tensor, x_gen: torch.Tensor, max_lag: int = 16, fig_size: tuple = (5, 4),
                          title: str = 'Autocorrelation real and generated data', feature_names: Optional[list] = None):
    """
    Returns a matplotlib plot of the autocorrelation for the given batch of samples.
    Args:
        x_real: torch.tensor of shape [n_samples, n_features, seq_len]
        x_gen: torch.tensor of shape [n_samples, n_features, seq_len]
        max_lag: maximum number of steps the autocorrelation is calculated for
        fig_size: tuple specifying the shape of the figure
        title: title of the plot
        feature_names: optional list of feature names

    """
    plt.style.use('seaborn')
    acor_real = acf(x_real, max_lag=max_lag).cpu().numpy()
    acor_gen = acf(x_gen, max_lag=max_lag).cpu().numpy()
    assert acor_real.shape == acor_gen.shape, "Autocorrelation of real and generated samples should have same shape"
    n_lags, n_features = acor_real.shape
    if feature_names is not None:
        assert len(feature_names) == n_features, "Number of different feature names must be equal to number of features"
    fig, ax = plt.subplots(n_features, 1, figsize=fig_size)
    lag = np.arange(0, n_lags)
    if n_features > 1:
        for i in range(n_features):
            ax[i].scatter(lag, acor_real[:, i], marker='x', label='real', color='b')
            ax[i].scatter(lag, acor_gen[:, i], marker='x', label='synthetic', color='r')
            ax[i].plot(lag, acor_real[:, i], color='b')
            ax[i].plot(lag, acor_gen[:, i], color='r')
            ax[i].set_xlabel("lag")
            ax[i].set_ylabel("correlation")
            if feature_names is not None:
                ax[i].set_title(feature_names[i])
    else:
        ax.scatter(lag, acor_real.squeeze(), marker='x', label='real', color='b')
        ax.scatter(lag, acor_gen.squeeze(), marker='x', label='synthetic', color='r')
        ax.plot(lag, acor_real.squeeze(), color='b')
        ax.plot(lag, acor_gen.squeeze(), color='r')
        ax.set_xlabel("lag")
        ax.set_ylabel("correlation")
    fig.suptitle(title)
    fig.tight_layout()
    plt.legend()

    return fig


if __name__ == '__main__':
    x = torch.randn(10, 1, 32)
    y = torch.randn(10, 1, 32)
    labels = torch.randint(0, 3, (10, 1))
    # fig = plot_acor_per_feature(x, y, max_lag=31)
    fig = plot_samples(x, labels=labels)
    plt.show()
