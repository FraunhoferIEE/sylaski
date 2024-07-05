import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
from syndatagenerators.metrics.mmd_score import new_mmd
import torch


## function to create input for mmd score (raw data array separated into days and without NaNs)
def df_to_day(df_raw):
    """
    Prepares raw data for Maximum Mean Discrepancy (MMD) calculation by separating the data into days and removing NaN values.

    Parameters:
    - df_raw: pandas DataFrame
        The raw data DataFrame containing time-series data.

    Returns:
    - data_into_days: list
        A list containing the data for each day separated into arrays without NaN values.

    Notes:
    - This function is used to preprocess raw data before calculating MMD scores.
    - It separates the data into individual days and removes NaN values by forward and backward filling.
    """

    print("preprocess raw data for MMD calculation...")
    data_into_days = []

    day = df_raw.index.day_of_year.astype(str)
    year = df_raw.index.year.astype(str)

    for column in df_raw.columns:
        df = pd.DataFrame(df_raw[column])
        df['day'] = [year[i] + day[i] for i in range(len(year))]
        df['time'] = df.index.time

        df = df.pivot(columns='day', index='time')

        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.dropna(axis=1, inplace=True)

        data_into_days.append(np.array(df.transpose()))

    return data_into_days


## function max mean discrepancy for clustering
def mmd_clustering_validation(labels, df_raw, plot_hist=True):
    """
    Performs clustering validation using Maximum Mean Discrepancy (MMD) score.

    Parameters:
    - labels: array-like
        The cluster labels assigned to each data point.
    - df_raw: pandas DataFrame
        The raw data DataFrame containing time-series data.
    - plot_hist: bool, optional (default=True)
        Whether to plot histograms of MMD scores for each cluster.

    Returns:
    - mmd_sample: list
        A list containing the MMD score per sample for each cluster.
    - mmd_sample_all: float
        The average MMD score per sample across all clusters.
    - mmd_mean: list
        A list containing the average MMD score for each cluster.
    - mmd_mean_all: float
        The average MMD score across all clusters.

    Notes:
    - This function calculates the MMD score for each pair of profiles within each cluster.
    - It optionally plots histograms of MMD scores for each cluster.
    """

    # input for mmd score
    df_days = df_to_day(df_raw)

    n_cluster = len(np.unique(labels))

    if plot_hist:
        # create subplots
        if n_cluster % 2 == 0:
            n_subplot = int(n_cluster / 2)
        else:
            n_subplot = int((n_cluster + 1) / 2)
        compare_samples = 100
        plt.subplots(n_subplot, 2, sharex=True)
        plt.suptitle('Max Mean Discrepancy of Profiles in Cluster')
        plt.xlabel('mmd')
        plt.xlim([0, 1])

    # mmd score for different clusters
    mmd_sample = list()
    mmd_mean = list()
    compare_samples = 100  # compare max 100 samples to save time

    for cluster in range(len(np.unique(labels))):
        print("calculating MMD for cluster " + str(cluster+1) + "/" + str(len(np.unique(labels))) + "...",
              end='\r', flush=True)
        current_cluster = np.array(df_days)[labels == cluster]
        if len(current_cluster) > compare_samples:
            current_cluster = current_cluster[random.sample(range(len(current_cluster)), compare_samples)]
        mmd_cluster = list()
        for profile1, profile2 in itertools.combinations(current_cluster, 2):
            mmd_cluster.append(new_mmd(torch.tensor(profile1), torch.tensor(profile2), 'cpu'))

        # plot histogram of mmd for each cluster
        if plot_hist:
            if n_cluster % 2 != 0:
                plt.subplot(n_subplot, 2, n_cluster + 1)
            if len(current_cluster) == 1:
                plt.subplot(n_subplot, 2, 1 + cluster)
                plt.title('cluster ' + str(cluster + 1) + ' (only one sample)')
                plt.grid()
            else:
                plt.subplot(n_subplot, 2, 1 + cluster)
                plt.grid()
                plt.hist(mmd_cluster, align='mid', bins=np.arange(0, 1, 0.05), edgecolor='black', color='skyblue')
                plt.title(
                    'cluster ' + str(cluster + 1) + ',  mean mmd: ' + str(round(np.mean(mmd_cluster), 2)) +
                    ', mmd per sample: ' + str(round(np.sum(mmd_cluster) / len(current_cluster), 2)))

        # store mmd values per cluster in list
        mmd_sample.append(np.sum(mmd_cluster) / len(current_cluster))  # mmd per sample for better comparability
        mmd_mean.append(np.mean(mmd_cluster))  # average of mmd
    # calculate average mmd value of all clusters (per sample and per mmd value)
    mmd_sample_all = np.nanmean(mmd_sample)
    mmd_mean_all = np.nanmean(mmd_mean)

    print()
    return mmd_sample, mmd_sample_all, mmd_mean, mmd_mean_all
