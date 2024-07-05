# %%
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from random import randint
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats


def energy_score(X_real, X_fake):
    axis = min(1, (X_real.ndim - 1))
    expectation_fake_real = np.mean(
        np.sqrt(np.sum(np.square(X_real-X_fake), axis=axis)))
    X_fake_shuffle = X_fake.copy()
    np.random.shuffle(X_fake_shuffle)
    expectation_fake_fake = np.mean(
        np.sqrt(np.sum(np.square(X_fake_shuffle-X_fake), axis=axis)))
    return expectation_fake_real - 0.5*expectation_fake_fake


def wasserstein_distance(X_real, X_fake):
    mu_r = np.mean(X_real, axis=0)
    cov_r = np.cov(X_real.T)
    mu_f = np.mean(X_fake, axis=0)
    cov_f = np.cov(X_fake.T)
    cov_f_cholesk = np.linalg.cholesky(cov_f)
    return np.sum(np.square(mu_f-mu_r)) + np.trace(cov_r+cov_f-2*np.linalg.cholesky(cov_f_cholesk.T@cov_r@cov_f_cholesk))


def rbf_kernel(x, y, sigma):
    return torch.exp((- squared_dist(x,y)**2)/(2*sigma**2))


def squared_dist(X, Y):
    # (x-y)'@(x-y) = x'@x - 2*x'@y + y'@y
    return (X*X).sum(1).reshape(-1, 1) - 2*X@Y.T + (Y*Y).sum(1).reshape(1, -1)


def rbf_kernel_mat(X, Y, sigma):
    return np.exp(- squared_dist(X, Y) / (2*np.square(sigma)))


def maximum_mean_discrepancy(X, Y, sigma=0.1, mod=False):
    n = X.shape[0]
    m = Y.shape[0]
    K1 = rbf_kernel_mat(X, X, sigma)
    if mod:
        K1 = K1-np.diag(np.diag(K1))
    term_xx = np.sum(K1)

    K2 = rbf_kernel_mat(X, Y, sigma)
    term_xy = np.sum(K2)

    K3 = rbf_kernel_mat(Y, Y, sigma)
    if mod:
        K3 = K3-np.diag(np.diag(K3))
    term_yy = np.sum(K3)

    if mod:
        return 1/(n*(n-1))*term_xx - 2/(n*m)*term_xy + 1/(m*(m-1))*term_yy
    return 1/(n*n)*term_xx - 2/(n*m)*term_xy + 1/(m*m)*term_yy


def mmd(x, y, device, bandwidth_range=[10, 15, 20, 50]):
    if(bandwidth_range is None):#use optimal bandwidth
        bandwidth_range = torch.pdist(torch.cat([x, y], dim=0)).median().unsqueeze(0).to(device)
    else:
        bandwidth_range = torch.tensor(bandwidth_range).to(device)
    # calc x^2 y^2 and xy
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    # get the squares and expand them to get a n x n matrix
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    # calc the the matrix part of the gaussian kernel
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * xy

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(yy.shape).to(device),
                  torch.zeros(xy.shape).to(device))
    # calc the kernel with diffrent bandwiths (maybe not needed)
    #bandwidth_range = [10, 15, 20, 50]
    for a in bandwidth_range:
        XX += torch.exp(-0.5*dxx/a)
        YY += torch.exp(-0.5*dyy/a)
        XY += torch.exp(-0.5*dxy/a)

    # take the mean and get result
    return torch.mean(XX + YY - 2. * XY).item()

def alt_mmd(x:torch.Tensor, y:torch.Tensor, device:str):
    dists = torch.cat([torch.pdist(x).flatten().cpu(), torch.pdist(y).flatten().cpu(), torch.cdist(x,y).flatten().cpu()])
    sigma = dists.median().unsqueeze(0).to(device)
    m = x.shape[0]
    n = y.shape[0]
    return (torch.sum(rbf_kernel(x,x,sigma))/(m*(m-1)) -  2 * torch.sum(rbf_kernel(x,y,sigma)) / (m*n) + torch.sum(rbf_kernel(y,y,sigma))/(n*(n-1))).item()


# %%

# Timeseries should be stationary ("mean, variance, and covariance shouldn’t change over time; by calculating the first-order difference")
def autocorrelation(Y_original, stride=1):
    if(stride == 0):
        return 1
    Y = Y_original[stride:]
    Y_shift = Y_original[:-stride]
    return np.corrcoef(Y, Y_shift)[0, 1]


def full_autocorrelation(Y, length=None):
    print("Better use from statsmodels.tsa.stattools import acf")
    if(length is None):
        length = Y.size - 1
    correlation = []
    for i in range(length):
        correlation.append(autocorrelation(Y, i))
    return correlation


def partial_autocorrelation(Y):
    print("Use from statsmodels.tsa.stattools import pacf")

def log_prob(gt, x):
    kde = stats.gaussian_kde(gt.T)
    e = kde.logpdf(x.T)
    return np.sum(e)


if(__name__ == "__main__"):
    Y = np.array([randint(1, 2) / 2 * i for i in range(100)])
    print(full_autocorrelation(Y, 41))
    print(acf(Y))

    # %%
    import numpy as np
    from matplotlib import pyplot as plt

    t = np.arange(0, 10)
    # print(t)
    D = np.abs(t.reshape(-1, 1)-t.reshape(1, -1))
    # print(D)

    def cov_func(d):
        return np.exp(-d**2/2**2) + 0.1*np.exp(-d/5)

    def cov_func2(d):
        return 0.1*np.exp(-d/10)

    S = cov_func(D)
    S2 = cov_func2(D)
    mu = np.zeros_like(t)
    gt = np.random.multivariate_normal(mu, S, size=(10,))
    x = np.random.multivariate_normal(mu, S, size=(10,))

    print("log-likelihood: " + str(log_prob(x, gt)))
    # %%
    dim = 1000
    n = 1000

    #t = np.arange(0, 1000)
    t = torch.arange(0, 1000)
    # print(t)
    #D = np.abs(t.reshape(-1,1)-t.reshape(1,-1))
    D = torch.abs(t.reshape(-1, 1)-t.reshape(1, -1))
    # print(D)

    def cov_func(d):
        return (np.exp(-d**2/2**2) + 0.1*np.exp(-d/5))/10

    def cov_func2(d):
        return 0.1*np.exp(-d/10)

    S = cov_func(D)
    S2 = cov_func2(D)

    x_mu = torch.zeros(dim)
    x_cov = torch.eye(dim)
    y_mu = torch.zeros(dim) + dim
    y_cov = dim/100 * torch.eye(dim) + dim/10
    # print(x_cov)
    # print(y_cov)
    px = MultivariateNormal(x_mu, x_cov)
    py = MultivariateNormal(y_mu, S)

    x = px.sample([n])
    y = py.sample([n])

    print(mmd(x, y))

# %%
