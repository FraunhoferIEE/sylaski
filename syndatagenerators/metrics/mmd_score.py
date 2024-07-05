import torch
import numpy as np


def mmd(x_real: torch.Tensor, x_fake: torch.Tensor):
    """
    Calculates the Maximum-Mean-Discrepancy between two tensors, using the (optimal) bandwidth calculated below;
    and a Gaussian kernel.
    Args:
        x_real: tensor of (real) data of shape [n_samples, n_features, seq_len]
        x_fake: tensor of (dake) data of shape [n_samples, n_features, seq_len]
    """
    n_1, f_1, seq_1 = x_real.shape
    n_2, f_2, seq_2 = x_fake.shape
    x_real = x_real.reshape(n_1, f_1 * seq_1).cpu()
    x_fake = x_fake.reshape(n_2, f_2 * seq_2).cpu()

    sigma = optim_bw_mmd(x_real, x_fake)

    xy = torch.cat([x_real, x_fake], dim=0)  # concatenate: new size [n_1+n_2,-1]
    distances = torch.cdist(xy, xy, p=2)  # pairwise distances in L^2
    k = torch.exp(
        -(distances ** 2) / (2 * sigma ** 2))  # + epsilon * torch.eye(n_1 + n_2)  # 2. for numerical stability
    k_x = k[:n_1, :n_1]
    k_y = k[n_1:, n_1:]
    k_xy = k[:n_1, n_1:]

    mmd_score = k_x.sum() / (n_1 * (n_1 - 1)) + k_y.sum() / (n_2 * (n_2 - 1)) - 2 * k_xy.sum() / (n_1 * n_2)

    return mmd_score


def optim_bw_mmd(x_real, x_fake):
    """
    Finds the optimal bandwidth for calculating the MMD score.
    """
    n_1 = len(x_real)
    n_2 = len(x_fake)
    x_real = x_real.view(n_1, -1)
    x_fake = x_fake.view(n_2, -1)

    distances = torch.pdist(torch.cat([x_real, x_fake], dim=0))
    indx = np.random.permutation(len(distances))[:500]
    sigma = distances[indx].median() / 2

    return sigma

def rbf_kernel_dist(dist, sigma):
    return torch.exp(- torch.square(dist) / (2*torch.square(sigma)))

def new_mmd(x:torch.Tensor, y:torch.Tensor, device:str):
    """
    Calculates the Maximum-Mean-Discrepancy between two tensors, using the (optimal) bandwidth calculated below;
    and a Gaussian kernel.
    Implemented as in 'A kernel method for the two-sample problem', Gretton et. al. (2012)
    Args:
        X: tensor of (real) data of shape [n1_samples, seq_len]
        Y: tensor of (fake) data of shape [n2_samples, seq_len]
        device: string of the device i.e. 'cpu' for cpu, 'cuda' for gpu, 'cuda:1' for gpu 2 etc.
    """
    x = x.to(device)
    y = y.to(device)
    x_dist = torch.pdist(x).flatten()
    y_dist = torch.pdist(y).flatten()
    xy_dist = torch.cdist(x,y).flatten()
    full_dists = torch.cat([x_dist, y_dist, xy_dist]).cpu()
    sigma = full_dists.median().unsqueeze(0).to(device)
    m = x.shape[0]
    n = y.shape[0]

    x_kernels = rbf_kernel_dist(x_dist, sigma)
    y_kernels = rbf_kernel_dist(y_dist, sigma)
    xy_kernels = rbf_kernel_dist(xy_dist, sigma)

    return (torch.sum(x_kernels)*2/(m*(m-1)) -  2 * torch.sum(xy_kernels) / (m*n) + torch.sum(y_kernels)*2/(n*(n-1))).item()