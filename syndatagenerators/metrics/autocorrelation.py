from typing import Tuple

import torch


def acf(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    """
    Computes the autocorrelation function for a given tensor and a given maximum number of time steps (lag).
    Args:
        x: input tensor of shape [n_samples, n_features, seq_len]
        max_lag: maximum lag for which the autocorrelation is computed
        dim:
    Returns:
        acor: torch.Tensor of shape [max_lag, n_features] representing autocorrelation per time step per feature.

    """
    acf_list = list()
    x_t = torch.swapaxes(x, 1, 2)
    x_c = x_t - x_t.mean((0, 1))
    std = torch.var(x_c, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x_c[:, i:] * x_c[:, :-i] if i > 0 else torch.pow(x_c, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        acor = torch.stack(acf_list)
        return acor
    else:
        acor = torch.cat(acf_list, 1)
        return acor
