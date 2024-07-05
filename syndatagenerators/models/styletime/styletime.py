import argparse

import torch as th
import torch.nn.functional as F

import matplotlib.pyplot as plt
import scipy.stats


'''
This mainly implements the functions described in the StyleTime paper
The operation and performance of the algorithm greatly depend on the used style and content functions so
changing these is the biggest factor in performance
'''
def content_score(y, y_c):
    x = th.arange(0, len(y_c),1)
    lin_regress = scipy.stats.linregress(x, y_c.detach().cpu())
    h = (x * th.tensor(lin_regress.slope) + th.tensor(lin_regress.intercept)).to(device=y.device, dtype=y.dtype)
    return F.mse_loss(y, h)

def log_difference(y_t_prime, y_t):
    return (th.log2(y_t_prime) - th.log2(y_t))

def log_differences(y):
    r = th.zeros(len(y) - 1).to(device=y.device, dtype=y.dtype)
    for t in range(len(y) - 1):
        r[t] = log_difference(y[t+1],y[t])
    return r

def average_log_return(y):
    running_sum = 0
    for t in range(len(y) - 1):
        running_sum += log_difference(y[t+1], y[t])
    return (running_sum / (len(y) - 1)).to(y.dtype)

def sample_auto_correlation(r):
    r_bar = r.mean()
    total_squared_err = F.mse_loss(r, th.fill(th.empty_like(r), r_bar))
    rho = th.zeros(len(r)).to(device=y.device, dtype=y.dtype)
    for tau in range(len(r)):
        for t in range(tau, len(r)):
            rho[tau] += (r[t] - r_bar) * (r[t-tau]-r_bar)
    return (rho / total_squared_err).to(r.dtype) 

def volatility(r):
    r_bar = r.mean()
    running_sum = 0
    for t in range(len(r)):
        running_sum += (r[t] - r_bar) ** 2
    return (running_sum / (len(r) - 1)).to(r.dtype)

def power_spectral_density(y):
    return th.view_as_real(th.fft.rfft(sample_auto_correlation(y)).mean())

def style_score(y, y_s, feats):
    loss = 0
    for f in feats:
        loss += F.mse_loss(f(y), f(y_s)).to(dtype=y.dtype)
    return loss

def total_variation_loss(y):
    loss = 0
    for t in range(len(y) - 1):
        loss += (y[t+1] - y[t]) ** 2
    return loss

def style_time(y, y_c, y_s, optimizer, alpha=1, beta=10, gamma=0.0001, I=250):
    for i in range(I):
        L_c = content_score(y, y_c)
        L_s = style_score(y, y_s, [lambda y: sample_auto_correlation(log_differences(y)), power_spectral_density, volatility])
        L_v = total_variation_loss(y)
        total_loss = alpha * L_c + beta * L_s + gamma * L_v
        print(f"loss {total_loss}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


if __name__ == "__main__":

    '''
    The training consists of applying the style and content functions, calculating the score and directly modifying the input sequence
    I.e backprop is applied to the INPUT instead of the weights/parameters of a ANN 
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--fileA', help='path to first ds')
    parser.add_argument('--fileB', help='path to first ds')
    parser.add_argument('--fig-path', default='out_fig.png', help='path for output figure')

    args = parser.parse_args()

    device = 'cuda' if th.cuda.is_available() else 'cpu'  

    summed_ds = th.load(args.fileA).to(device)
    concat_ds = th.load(args.fileB).to(device)

    s_idx = 15
    summed_wave = summed_ds[s_idx].squeeze().float() + 2
    concat_wave = concat_ds[s_idx].squeeze().float() + 2

    y = th.nn.Parameter(summed_wave.clone())
    #concat_wave.requires_grad_(True)
    optimizer = th.optim.RMSprop([y], 0.001)
    style_time(y, summed_wave, concat_wave, optimizer)


    # save output
    fig, axs = plt.subplots(1,2)
    axs[0].plot(summed_wave.detach())
    axs[1].plot(y.detach())

    fig.savefig(args.fig_path)
