# performing clustering on load profiles

## import modules
import pandas as pd
import syndatagenerators.data_analysis.clustering as clustering
import syndatagenerators.data_analysis.feature_extraction as feature_extraction
from tslearn.clustering import TimeSeriesKMeans
from sklearn.decomposition import PCA


## data
# load_profiles = path of csv file with time steps in first column and profiles in following columns

# function to prepare dataFrame and dimension reduced data for clustering
def prepare_data(load_profiles, n_pca=5):
    """
    Prepares data for clustering by performing dimension reduction and removing profiles with NaNs.

    Parameters:
    - load_profiles: str
        The path of the CSV file containing time steps in the first column and profiles in the following columns.
    - n_pca: int, optional (default=5)
        The number of principal components to use for PCA dimension reduction.

    Returns:
    - df_av_cl: pandas DataFrame
        The dimension-reduced data for clustering with NaN profiles removed.
    - df_av_pca: pandas DataFrame
        The dimension-reduced data using PCA.
    - df: pandas DataFrame
        The original DataFrame containing the raw data.

    Notes:
    - This function loads the data from the CSV file, sets the datetime index, and removes the time column.
    - It then performs dimension reduction to average weeks and removes profiles with NaN values.
    - Lastly, it performs dimension reduction to n_pca components using PCA.
    """

    # load data, set datetime index and delete time column
    if '.zip' in load_profiles:
        df = pd.read_csv(load_profiles, compression='zip')
    else:
        df = pd.read_csv(load_profiles)
    df.index = pd.to_datetime(df.iloc[:, 0])
    df.drop(df.columns[0], axis=1, inplace=True)

    # remove possible duplicated indices
    df = df[~df.index.duplicated(keep='first')]

    # dimension reduction to average weeks and remove profiles with NaNs
    df_av = feature_extraction.average_weeks(df)
    nan_id = []
    for column in df_av.columns:
        if any(df_av[column].isna()):
            nan_id.append(column)
    df_av.drop(columns=nan_id, inplace=True)
    df.drop(columns=nan_id, inplace=True)

    # dimension reduction to n_pca components with PCA
    df_av_pca = pd.DataFrame(PCA(n_components=n_pca).fit_transform(df_av.T))

    # transform data for clustering algorithm
    df_av_cl = df_av.T
    df_av_cl.columns = range(len(df_av_cl.columns))  # remove multi index

    return df_av_cl, df_av_pca, df


# function to run clustering (default with 5 PCA components, kmeans)
def run_clustering(load_profiles, n_clusters, n_pca=5, calc_mmd=False):
    """
    Runs clustering on the provided data using k-means algorithm with default settings.

    Parameters:
    - load_profiles: str
        The path of the CSV file containing time steps in the first column and profiles in the following columns.
    - n_clusters: int
        The number of clusters to create.
    - n_pca: int, optional (default=5)
        The number of principal components to use for PCA dimension reduction.
    - calc_mmd: bool, optional (default=False)
        Whether to calculate the Maximum Mean Discrepancy (MMD) for clustering validation.

    Returns:
    - labels: array-like
        The cluster labels assigned to each data point.
    - clustering_obj: object
        The clustering object containing the results and visualization.

    Notes:
    - This function prepares the data by performing dimension reduction and removing profiles with NaN values.
    - It then runs clustering using the k-means algorithm with default settings.
    - If calc_mmd is set to True, it calculates the MMD for clustering validation and plots the results.
    - If calc_mmd is set to False, it only plots the cluster visualization without MMD calculation.
    """

    df_av_cl, df_av_pca, df = prepare_data(load_profiles, n_pca)

    # initialize clustering model
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric='euclidean', init='k-means++',
                              max_iter=100, n_init=10)

    # initialize clustering object and execute clustering
    clustering_obj = clustering.Clustering(kmeans, df_av_pca)
    labels = clustering_obj.clustering(repetition=100)

    # validation and plotting
    if calc_mmd:  # calculation of mmd may take a while
        clustering_obj.plot_cluster(x_plot=df_av_cl, x_reference=df_av_cl, df_raw=df)
    else:
        clustering_obj.plot_cluster(x_plot=df_av_cl, x_reference=df_av_cl)

    return labels, clustering_obj
