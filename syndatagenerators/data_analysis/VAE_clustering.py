# Variational Autoencoder as dimension reduction technique for load profile clustering

## import modules
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import syndatagenerators.data_analysis.mmd_score_clustering as mmd_score
import syndatagenerators.data_analysis.clustering as clustering


## create class for preparation of input data (right format)
class AvDataset(Dataset):
    """
    Dataset class for preparing input data in the right format for average load profiles.

    Parameters:
    - df_av: DataFrame
        DataFrame containing average load profiles.

    Attributes:
    - data: torch.Tensor
        Tensor containing the input data in the right format.
    - n_samples: int
        Number of samples in the dataset.

    Methods:
    - __getitem__(self, item):
        Retrieves a sample from the dataset at the specified index.
    - __len__(self):
        Returns the total number of samples in the dataset.
    """

    def __init__(self, df_av):
        self.data = torch.tensor(df_av.values.astype(np.float32)).T
        self.n_samples = self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n_samples


class RawDataset(Dataset):
    """
    Dataset class for preparing input data in the right format for raw load profiles.

    Parameters:
    - df_raw: DataFrame
        DataFrame containing raw load profiles.

    Attributes:
    - data: torch.Tensor
        Tensor containing the input data in the right format.
    - n_samples: int
        Number of samples in the dataset.

    Methods:
    - __getitem__(self, item):
        Retrieves a sample from the dataset at the specified index.
    - __len__(self):
        Returns the total number of samples in the dataset.
    """

    def __init__(self, df_raw):
        self.data = torch.tensor(df_raw.values.astype(np.float32)).T
        self.n_samples = self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n_samples


# create dataset instance
dataset = AvDataset()

# create mini-batches for training
loader = DataLoader(
    dataset=dataset,
    batch_size=10,
    shuffle=True,
    drop_last=True
)


## create Encoder, Decoder and Variational Autoencoder (VAE)
# encoder
class Encoder(nn.Module):
    """
    Encoder class for the variational autoencoder (VAE) architecture.

    Parameters:
    - n_features: int
        Number of input features.
    - n_hidden: int, optional (default=128)
        Number of units in the hidden layers.
    - n_latent: int, optional (default=32)
        Dimensionality of the latent space.

    Attributes:
    - n_latent: int
        Dimensionality of the latent space.
    - layers: nn.Sequential
        Sequential module containing the layers of the encoder.

    Methods:
    - forward(self, x):
        Forward pass of the encoder, encoding input data into the latent space.
    - latent_dist(self, x):
        Computes the mean and standard deviation of the latent distribution without noise.
    """

    def __init__(self, n_features, n_hidden=128, n_latent=32):
        super().__init__()
        self.n_latent = n_latent
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.LeakyReLU(),  # set negative values to small non-negative values (activation function)
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_latent * 2)
        )

    # encoding process
    def forward(self, x):
        """
        Forward pass of the encoder.

        Parameters:
        - x: torch.Tensor
            Input data to be encoded.

        Returns:
        - latent: torch.Tensor
            Encoded representation (latent variable) of the input data.
        """

        layers_out = self.layers(x)
        mu = layers_out[:, :self.n_latent]
        log_var = layers_out[:, self.n_latent:]  # logarithm of variance of latent variable distribution
        noise = torch.randn_like(mu)
        std = torch.exp(0.5 * log_var)
        latent = mu + std * noise
        self.kl_loss = 0.5 * torch.sum(torch.exp(log_var) + mu ** 2 - 1 - log_var)
        return latent

    # mean and standard deviation of latent distribution without noise
    def latent_dist(self, x):
        """
        Computes the mean and standard deviation of the latent distribution without noise.

        Parameters:
        - x: torch.Tensor
            Input data.

        Returns:
        - mu: torch.Tensor
            Mean of the latent distribution.
        - std: torch.Tensor
            Standard deviation of the latent distribution.
        """

        layers_out = self.layers(x)
        mu = layers_out[:, :self.n_latent]
        log_var = layers_out[:, self.n_latent:]
        std = torch.exp(0.5 * log_var)
        return mu, std


# decoder
class Decoder(nn.Module):
    """
    Decoder class for the variational autoencoder (VAE) architecture.

    Parameters:
    - n_features: int
        Number of output features.
    - n_hidden: int, optional (default=128)
        Number of units in the hidden layers.
    - n_latent: int, optional (default=32)
        Dimensionality of the latent space.

    Attributes:
    - n_latent: int
        Dimensionality of the latent space.
    - layers: nn.Sequential
        Sequential module containing the layers of the decoder.

    Methods:
    - forward(self, latent):
        Forward pass of the decoder, reconstructing input data from the latent space.
    """

    def __init__(self, n_features, n_hidden=128, n_latent=32):
        super().__init__()
        self.n_latent = n_latent
        self.layers = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_features)
        )

    def forward(self, latent):
        """
          Forward pass of the decoder.

          Parameters:
          - latent: torch.Tensor
              Latent representation of input data.

          Returns:
          - reconstructed: torch.Tensor
              Reconstructed data from the latent space.
          """

        return self.layers(latent)


# variational autoencoder
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) architecture composed of an encoder and a decoder.

    Parameters:
    - n_features: int
        Number of input features.
    - n_hidden: int, optional (default=128)
        Number of units in the hidden layers of the encoder and decoder.
    - n_latent: int, optional (default=32)
        Dimensionality of the latent space.

    Attributes:
    - encoder: Encoder
        Instance of the Encoder class responsible for encoding input data into the latent space.
    - decoder: Decoder
        Instance of the Decoder class responsible for reconstructing input data from the latent space.

    Methods:
    - forward(self, x):
        Forward pass of the VAE, encoding input data into the latent space and reconstructing it.
    """

    def __init__(self, n_features, n_hidden=128, n_latent=32):
        super().__init__()
        self.encoder = Encoder(n_features, n_hidden, n_latent)
        self.decoder = Decoder(n_features, n_hidden, n_latent)

    def forward(self, x):
        """
        Forward pass of the VAE.

        Parameters:
        - x: torch.Tensor
            Input data to be encoded and reconstructed.

        Returns:
        - reconstructed: torch.Tensor
            Reconstructed data from the latent space.
        """

        latent = self.encoder(x)
        return self.decoder(latent)


## train model
n_features = dataset.data.shape[1]
mdl = VAE(n_features)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mdl.parameters())

n_epochs = 100
kl_weight = 1e-6
loss_list = []
kl_loss_list = []

for epoch in range(n_epochs):
    for x in loader:
        recon = mdl(x)
        kl_loss = mdl.encoder.kl_loss
        loss = criterion(recon, x) + kl_loss * kl_weight
        kl_loss_list.append(kl_loss.item())
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f'{epoch =}, {loss.item() =}')

## plot reconstruction loss and KL-divergence
plt.subplots(1, 2, figsize=[18, 6])
plt.subplot(1, 2, 1, facecolor='snow')
plt.plot(loss_list)
plt.xlabel('number of epochs x batch size')
plt.grid()
plt.title('reconstruction loss')
plt.xlabel('iteration')
plt.subplot(1, 2, 2, facecolor='snow')
plt.plot(kl_loss_list)
plt.title('KL-divergence')
plt.grid()
plt.xlabel('number of epochs x batch size')

## generate synthetic data with decoder
samples = 3
plt_latent = torch.randn(samples, mdl.decoder.n_latent)
fake = mdl.decoder(plt_latent)
plt.figure(figsize=[12, 8])
plt.plot(fake.detach().T)
ax = plt.gca()
ax.set_facecolor('snow')
plt.grid()
plt.xlabel('time step')
plt.ylabel('normalized power [kWh]')
plt.title('synthetic load profiles generated by decoder')

## visualize latent-space-distribution of load profile (sample)
sample = 1
z_list = []
for k in range(100):
    z = mdl.encoder(x[sample].view(1, -1))
    z_list.append(list(z.detach().numpy()))
z_mat = np.array(z_list).squeeze()

fig = plt.figure(figsize=[12, 8])
plt.plot(z_mat.T, '.')
ax = plt.gca()
ax.set_facecolor('snow')
plt.title('latent-space-distribution of profile ' + str(sample))
plt.grid(axis='x', which='both')
plt.minorticks_on()
plt.xlabel('dimension of latent space')

## visualize mean and standard deviation of latent variables from encoder network
mu, std = mdl.encoder.latent_dist(x)
plt.subplots(4, 4)
plt.suptitle('mean and standard deviation of latent variables (encoder)')
for k in range(16):
    plt.subplot(4, 4, k + 1, facecolor='snow')
    plt.plot(mu[k].detach(), 'ks')
    plt.plot(mu[k].detach() + 2 * std[k].detach(), '+', markeredgecolor='dimgrey')
    plt.plot(mu[k].detach() - 2 * std[k].detach(), '+', markeredgecolor='dimgrey')
    plt.grid(which='both', color='lightgrey')
    plt.minorticks_on()

## generate synthetic data with decoder (different data for one profile)
latent = torch.randn(1, mdl.decoder.n_latent)
plt.figure()
for k in range(4):
    latent[0, 10] = -2 + 4 * k / 3  # -2 to 2 in steps of 4/3
    fake = mdl.decoder(latent)
    plt.plot(fake.detach().T)
plt.grid()

## validation with mmd
training_data = dataset.data.numpy()
latent = torch.randn(1000, mdl.decoder.n_latent)
fake_data = mdl.decoder(latent).detach().numpy()

mmd = mmd_score.mmd(fake_data, training_data)
print(f'MMD: {mmd:.4f}')


## clustering with mean of latent distribution

# Obtain the latent variables (mean and standard deviation) for all input samples using the VAE encoder.
mu, std = mdl.encoder.latent_dist(dataset.data)
mu_cl = mu.detach().numpy()  # Convert mean tensor to a numpy array.
std_cl = std.detach().numpy()  # Convert standard deviation tensor to a numpy array.

# Initialize the clustering model.
kmeans = TimeSeriesKMeans(n_clusters=11, metric='euclidean', init='k-means++',
                          max_iter=100, n_init=10, random_state=0)

# Perform clustering using the mean of the latent distribution obtained from the encoder.
c = clustering.Clustering(kmeans, mu_cl)  # Initialize the clustering object with the KMeans model and latent means.
labels = c.clustering()  # Obtain cluster labels for the data.

# Plot the clusters based on the latent variables and show scores if specified.
c.plot_cluster(x_plot=dataset.data, x_reference=dataset.data, show_scores=True)

## validate clustering with mmd_score

# Validate the clustering using the maximum mean discrepancy (MMD) score.
mmd_sample, mmd_mean = c.mmd_validation(c)  # Compute MMD scores for the clusters.
