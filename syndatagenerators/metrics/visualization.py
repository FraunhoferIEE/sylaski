import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def t_SNE(x_real, x_fake, feature_mean=True, perplexity=80, verbose=True):
    """
    Computes the two dimensional t-SNE embedding of the two input tensors.
    Args:
        x_real: torch.tensor of shape [n_samples, n_features, seq_len]
        x_fake: torch.tensor of shape [n_samples, n_features, seq_len]
        feature_mean: boolean stating whether the mean of the feature dimension shall be calculated, or if the axis shall
                        be just reshaped.
        perplexity: perplexity of the t-SNE embedding.
    Returns:
        x_real_embedded: numpy array of embedded first tensor
        x_fake_embedded: numpy array of embedded second tensor
    """
    x_real = x_real.cpu().numpy()
    x_fake = x_fake.cpu().numpy()
    assert x_real.shape[-1] == x_fake.shape[-1], "the two input tensors need to have the same sequence length"

    samples, dim, seq_len = x_real.shape
    if feature_mean:
        for i in range(samples):
            if i == 0:
                prep_data = np.reshape(np.mean(x_real[0, :, :], 0), [1, seq_len])
                prep_data_hat = np.reshape(np.mean(x_fake[0, :, :], 0), [1, seq_len])
            else:

                prep_data = np.concatenate((prep_data,
                                            np.reshape(np.mean(x_real[i, :, :], 0), [1, seq_len])))
                prep_data_hat = np.concatenate((prep_data_hat,
                                                np.reshape(np.mean(x_fake[i, :, :], 0), [1, seq_len])))

    else:
        prep_data = x_real.reshape(-1, seq_len)
        prep_data_hat = x_fake.reshape(-1, seq_len)

    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

    tsne = TSNE(n_components=2, verbose=1 if verbose else 0, perplexity=perplexity, n_iter=1000)
    tsne_embedding = tsne.fit_transform(prep_data_final)

    indx = len(prep_data)
    x_real_embedded = tsne_embedding[:indx]
    x_fake_embedded = tsne_embedding[indx:]

    return x_real_embedded, x_fake_embedded


def plot_TSNE(x_real, x_fake, feature_mean=True, perplexity=80, verbose=False, use_seaborn=False, alpha=1):
    import matplotlib.pyplot as plt
    
    if use_seaborn:
        import seaborn as sns
        sns.set()
    
    embed_real, embed_fake = t_SNE(x_real, x_fake, feature_mean, perplexity, verbose)
        
    fig = plt.figure()
    plt.scatter(embed_real[:,0], embed_real[:,1], marker='.', label="Real Data", alpha=alpha)
    plt.scatter(embed_fake[:,0], embed_fake[:,1], marker='.', label="Fake Data", alpha=alpha)
    plt.legend()
    return fig