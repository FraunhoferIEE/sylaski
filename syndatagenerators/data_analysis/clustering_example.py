# cluster analysis of load profiles

# Import necessary modules
from syndatagenerators.data_analysis.clustering import Clustering  # Import the Clustering class
from syndatagenerators.data_analysis.clustering_run import prepare_data, run_clustering  # Import helper functions
from tslearn.clustering import TimeSeriesKMeans  # Import TimeSeriesKMeans for time series clustering

## path to data (.csv)
load_profile_data = r"C:\Users\lriedl\Documents\1_Projekte\SyLas-KI\Daten\OM_subset_Final.csv"

## clustering data with best approach (use PCs of averaged weeks)

# Define the number of clusters for clustering
n_clusters = 18

# Run clustering using the best approach (using principal components of averaged weeks) and calculate MMD
labels, obj = run_clustering(load_profile_data, n_clusters, n_pca=5, calc_mmd=True)

## clustering data with individual approach

# Get averaged and raw data in suitable format for clustering
df_averaged_load_data, _, df_raw_data = prepare_data(load_profile_data)

# Initialize clustering model (K-Means for example)
cluster_number_placeholder = 2  # Placeholder for initial cluster number
kmeans = TimeSeriesKMeans(n_clusters=cluster_number_placeholder, metric='euclidean', init='k-means++',
                          max_iter=100, n_init=10)

# Initialize clustering object with the KMeans model and averaged load data
clustering_input = df_averaged_load_data  # Use averaged data for clustering for example
clustering_obj = Clustering(kmeans, df_averaged_load_data)

# Determine the best option for cluster number
clustering_obj.cluster_number()

# Set new cluster number based on analysis or preference
n_clusters = 12

# Update the number of clusters in the KMeans model
kmeans.n_clusters = n_clusters

# Run clustering with the updated number of clusters
labels = clustering_obj.clustering(repetition=100)  # Perform clustering with 100 repetitions for example

# Plot clustering results
# Averaged data is recommended for visualization
clustering_obj.plot_cluster(x_plot=df_averaged_load_data, x_reference=df_averaged_load_data, df_raw=df_raw_data)
