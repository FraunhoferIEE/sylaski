class Clustering:
    """
    A class for performing clustering analysis.

    Parameters:
    -----------
    model : object
        The clustering model to be used. It should be compatible with scikit-learn or tslearn.
    x_input : array-like
        The input data for clustering. Each row represents a load profile, with each column corresponding to a time step.

    Attributes:
    -----------
    model : object
        The clustering model.
    x_input : array-like
        The input data for clustering.
    labels : array-like or None
        The cluster labels assigned by the clustering algorithm.
    n_cluster : int
        The number of clusters.
    scores : dict or None
        Dictionary containing validation scores such as Silhouette Score, Calinski-Harabasz Score,
        Davies-Bouldin Index, and Sum Squared Error.
    """

    def __init__(self, model, x_input):
        self.model = model  # Initializing the clustering model
        self.x_input = x_input  # Input data for clustering
        self.labels = None  # Initialized as None, later stores the cluster labels
        # Determining the number of clusters based on the type of clustering model
        if self.model.__class__.__name__ == 'GaussianMixture':
            self.n_cluster = self.model.n_components
        else:
            self.n_cluster = self.model.n_clusters
        self.scores = None  # Initialized as None, later stores the clustering validation scores

    def __str__(self):
        if self.labels is None:
            label_str = '[]'
        else:
            label_str = str(self.labels.shape)
        return ('Clustering Object\n' + 'model: ' + self.model.__class__.__name__ +
                '\ninput: ' + str(self.x_input.shape) +
                '\nmetric: ' + self.model.metric + '\ncluster: ' + str(self.n_cluster) +
                '\nlabels: ' + label_str +
                '\nscores: ' + str(self.scores))

    # plot different scores to estimate optimal number of clusters
    def cluster_number(self, try_number=None):
        """
        Analyzes different clustering evaluation metrics for a range of cluster numbers and visualizes the results.

        Parameters:
        -----------
        try_number : int or None, optional (default=None)
            The maximum number of clusters to try. If None, the range will be from 2 to 17.

        Notes:
        ------
        This method calculates and visualizes the following clustering evaluation metrics:
            - Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters.
            - Calinski-Harabasz Score: Evaluates cluster validity based on the ratio of between-cluster dispersion
              and within-cluster dispersion.
            - Davies-Bouldin Index: Provides information about the average 'similarity' between each cluster and its
              most similar one, based on centroids and distances.
            - Inertia (SSE): Sum of squared distances of samples to their closest cluster center.
              Applicable for KMeans and other centroid-based algorithms.

        Returns:
        --------
        Visualization of evaluation metrics in dependence of the cluster number.

        """

        from tslearn.clustering import silhouette_score
        from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
        import matplotlib.pyplot as plt

        s_score = []  # Silhouette Score
        ch_score = []  # Calinski-Harabasz score (variance ratio criterion)
        db_score = []  # Davies-Boulding Index
        sse_score = []  # sum squared error

        # calculate scores for different number of clusters (2 - try_number)
        if try_number is None:
            try_cluster = range(2, 18)
        else:
            try_cluster = range(2, try_number)

        for n_cluster in try_cluster:
            if self.model.__class__.__name__ == 'GaussianMixture':
                self.model.n_components = n_cluster
                labels = self.model.fit_predict(self.x_input)
            else:
                self.model.n_clusters = n_cluster
                self.model.fit_predict(self.x_input)
                labels = self.model.labels_
            db_score.append(davies_bouldin_score(self.x_input, labels))
            s_score.append(silhouette_score(self.x_input, labels, metric='euclidean'))
            ch_score.append(calinski_harabasz_score(self.x_input, labels))
            if self.model.__class__.__name__ not in ['AgglomerativeClustering', 'GaussianMixture']:
                sse_score.append(self.model.inertia_)

        # plot
        c = 'palevioletred'
        mc = 'deeppink'
        plt.subplots(4, 1, sharex=True, figsize=[10, 12])
        plt.subplot(4, 1, 1)
        plt.plot(try_cluster, s_score, marker='.', markerfacecolor=mc, color=c)
        plt.title("Silhouette Score")
        plt.grid()
        plt.subplot(4, 1, 2)
        plt.plot(try_cluster, ch_score, marker='.', markerfacecolor=mc, color=c)
        plt.title("Calinski-Harabasz Score")
        plt.grid()
        plt.subplot(4, 1, 3)
        plt.plot(try_cluster, db_score, marker='.', markerfacecolor=mc, color=c)
        plt.title("Davies-Boulding Index")
        plt.grid()
        if self.model.__class__.__name__ not in ['AgglomerativeClustering', 'GaussianMixture']:
            plt.subplot(4, 1, 4)
            plt.plot(try_cluster, sse_score, marker='.', markerfacecolor=mc, color=c)
            plt.xlabel("Number of Clusters")
            plt.title("Inertia (SSE)")
            plt.grid()
        plt.suptitle(
            'Validation Clustering (' + self.model.__class__.__name__ + ', ' + self.model.init + ', ' +
            self.model.metric + ')'
        )
        plt.minorticks_on()
        plt.show()

    # plot dendrogram (visualize cluster and possible number of cluster)
    def plot_dendrogram(self):
        """
        Plot dendrogram to visualize clustering and possible number of clusters.

        Notes:
        ------
        This method generates a dendrogram plot using hierarchical clustering. It computes the linkage
        matrix based on the input data and plots the dendrogram to visualize the hierarchical clustering
        structure. The dendrogram helps in understanding the clustering patterns and determining the
        possible number of clusters.
        """

        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage

        linkage_data = linkage(self.x_input, method='ward', metric='euclidean')
        fig, ax = plt.subplots(1, figsize=[20, 10])
        dendrogram(linkage_data)
        ax.set_xticklabels([])
        plt.grid(axis='y')
        plt.title('dendrogram (wards method, euclidean metric)')
        plt.tight_layout()

    # clustering
    def clustering(self, repetition=1):
        """
        Perform clustering on the input data multiple times and return the best set of cluster labels.

        Parameters:
        -----------
        repetition : int, optional (default=1)
            The number of times clustering should be performed.

        Returns:
        --------
        labels : array-like
            The best set of cluster labels obtained after multiple clustering runs.

        Notes:
        ------
        This method performs clustering on the input data `repetition` times and evaluates each set of cluster
        labels using a validation metric. The best set of cluster labels is determined based on the evaluation
        metric. The method updates the `labels` attribute of the instance with the best labels and returns them.
        """

        labels_temp = []
        scores = []
        for run in range(repetition):
            print("clustering run " + str(run + 1) + "/" + str(repetition) + "...", end='\r', flush=True)
            self.labels = self.model.fit_predict(self.x_input)
            labels_temp.append(self.labels)
            scores_temp = self.validation()
            scores.append(scores_temp[0] + (scores_temp[1] / 10) - scores_temp[2])
        print()
        best_score_position = scores.index(max(scores))
        labels = labels_temp[best_score_position]
        self.labels = labels
        self.model.labels_ = labels
        self.scores = None
        return labels

    # clustering validation
    def validation(self, x_reference=None, df_raw=None):
        """
        Evaluate the quality of clustering using various validation metrics.

        Parameters:
        -----------
        x_reference : array-like or None, optional (default=None)
            Reference data to compare clustering results with. If None, the input data used for clustering will be used.
        df_raw : DataFrame or None, optional (default=None)
            Raw data for additional validation metrics. If provided, the method will calculate the Maximum Mean Discrepancy (MMD).

        Returns:
        --------
        result : list
            A list containing the computed validation scores in the following order:
                1. Silhouette Score
                2. Calinski-Harabasz Score
                3. Davies-Bouldin Index
                4. Sum Squared Error (SSE)
            If `df_raw` is provided, the list will also include the MMD score.

        Notes:
        ------
        This method evaluates the quality of clustering using the following validation metrics:
            - Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters.
            - Calinski-Harabasz Score: Evaluates cluster validity based on the ratio of between-cluster dispersion
              and within-cluster dispersion.
            - Davies-Bouldin Index: Provides information about the average 'similarity' between each cluster and its
              most similar one, based on centroids and distances.
            - Sum Squared Error (SSE): Sum of squared distances of samples to their closest cluster center.

        If `df_raw` is provided, the method also calculates the Maximum Mean Discrepancy (MMD) between the clustering
        labels and the raw data.
        """

        import numpy as np
        from tslearn.clustering import silhouette_score
        from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
        from scipy.spatial.distance import euclidean

        if self.labels is None:
            self.clustering()

        if x_reference is None:
            x_reference = self.x_input

        # Silhouette Score
        s_score = silhouette_score(x_reference, self.labels, metric='euclidean')
        # Calinski-Harabasz score (variance ratio criterion)
        ch_score = calinski_harabasz_score(x_reference, self.labels)
        # Davies-Boulding Index
        db_score = davies_bouldin_score(x_reference, self.labels)
        # Sum Squared Error
        sse_score = 0.0
        for label in np.unique(self.labels):
            cluster_time_series = self.x_input.loc[self.labels == label]
            cluster_center = cluster_time_series.mean(axis=0)
            distances = [euclidean(ts, cluster_center) for ts in cluster_time_series.values]
            sse_score += np.sum(np.array(distances) ** 2)
        sse_score = sse_score / len(np.unique(self.labels))
        self.scores = {
            "Silhouette Score": s_score,
            "Calinski-Harabasz Score": ch_score,
            "Davies-Bouldin Index": db_score,
            "SSE": sse_score
        }
        result = [s_score, ch_score, db_score, sse_score]

        if df_raw is not None:
            mmd_sample = self.mmd_validation(df_raw=df_raw, plot_hist=False)
            self.scores = {
                "Silhouette Score": s_score,
                "Calinski-Harabasz Score": ch_score,
                "Davies-Bouldin Index": db_score,
                "SSE": sse_score,
                "MMD": mmd_sample
            }
            result.append(mmd_sample)

        return result

    # plot clustering results
    def plot_cluster(self, x_plot=None, x_reference=None, calculate_scores=True, df_raw=None):
        """
         Plot clusters and their centroids based on the clustering results.

         Parameters:
         -----------
         x_plot : array-like or None, optional (default=None)
             Data to be plotted. If None, the input data used for clustering will be plotted.
         x_reference : array-like or None, optional (default=None)
             Reference data for validation. If provided, clustering quality will be evaluated based on this data.
         calculate_scores : bool, optional (default=True)
             Whether to calculate validation scores for the clustering. If True, validation scores will be calculated and displayed.
         df_raw : DataFrame or None, optional (default=None)
             Raw data for additional validation metrics. If provided, the method will calculate the Maximum Mean Discrepancy (MMD).

         Returns:
         --------
         fig : Figure
             The matplotlib figure object containing the plotted clusters.

         Notes:
         ------
         This method plots the clusters and their centroids based on the clustering results. If validation is requested,
         the method calculates and displays validation scores, including Silhouette Score, Calinski-Harabasz Score,
         Davies-Bouldin Index, and Sum Squared Error (SSE). If additional raw data is provided, the Maximum Mean Discrepancy (MMD)
         between the clustering labels and the raw data will also be calculated.
         """

        import matplotlib.pyplot as plt
        import numpy as np

        if self.labels is None:
            self.clustering()

        if x_plot is None:
            x_plot = self.x_input

        # validation
        if self.scores is None:
            if calculate_scores:
                if x_reference is None:
                    x_reference = self.x_input

                validation = self.validation(x_reference)
                title = ('Load Profile Clustering (' + self.model.__class__.__name__ +
                         ', ' + self.model.init + ', ' + self.model.metric + ')\n' +
                         'Silhouette = ' + str(np.round(validation[0], 2)) +
                         ', Calinski-Harabasz = ' + str(np.round(validation[1], 2)) +
                         ', Davies-Bouldin = ' + str(np.round(validation[2], 2)) +
                         ', SSE = ' + str(np.round(validation[3], 2)))
                if df_raw is not None:
                    validation = self.validation(x_reference, df_raw)
                    mmd_score = np.array(validation[4])
                    title = (
                            'Load Profile Clustering (' + self.model.__class__.__name__ +
                            ', ' + self.model.init + ', ' + self.model.metric + ')\n' +
                            'Silhouette = ' + str(np.round(validation[0], 2)) +
                            ', Calinski-Harabasz = ' + str(np.round(validation[1], 2)) +
                            ', Davies-Bouldin = ' + str(np.round(validation[2], 2)) +
                            ', SSE = ' + str(np.round(validation[3], 2)) +
                            ', MMD = ' + str(np.round(np.mean(mmd_score[mmd_score > 0]), 2)))
        else:
            if df_raw is not None:
                mmd_score = np.array(self.scores.get('MMD'))
                title = ('Load Profile Clustering (' + self.model.__class__.__name__ + ', ' + self.model.init + ', ' +
                         self.model.metric + ')\n' +
                         'Silhouette = ' + str(np.round(self.scores.get('Silhouette Score'), 2)) +
                         ', Calinski-Harabasz = ' + str(np.round(self.scores.get('Calinski-Harabasz Score'), 2)) +
                         ', Davies-Bouldin = ' + str(np.round(self.scores.get('Davies-Bouldin Index'), 2)) +
                         ', SSE = ' + str(np.round(self.scores.get('SSE'), 2)) +
                         ', MMD = ' + str(np.round(np.mean(mmd_score[mmd_score > 0]), 2)))
            else:
                title = ('Load Profile Clustering (' + self.model.__class__.__name__ + ', ' + self.model.init + ', ' +
                         self.model.metric + ')\n' +
                         'Silhouette = ' + str(np.round(self.scores.get('Silhouette Score'), 2)) +
                         ', Calinski-Harabasz = ' + str(np.round(self.scores.get('Calinski-Harabasz Score'), 2)) +
                         ', Davies-Bouldin = ' + str(np.round(self.scores.get('Davies-Bouldin Index'), 2)) +
                         ', SSE = ' + str(np.round(self.scores.get('SSE'), 2)))

        # plot
        if self.n_cluster % 2 == 0:
            n_subplot = int(self.n_cluster / 2)
        else:
            n_subplot = int((self.n_cluster + 1) / 2)

        fig = plt.subplots(n_subplot, 2, sharex=True, figsize=[18, 10])
        x = range(48, 96 * 21, 96)
        # x = range(24, 48 * 21, 48)  # if data is in 30-minute steps
        # plt.xlim([0, len(self.x_input)])
        # plt.ylim([0, 1])
        for cluster in range(self.n_cluster):
            plt.subplot(n_subplot, 2, 1 + cluster)
            plt.xticks(x, ['Monday', 'Tuesday', 'Wednesday',
                           'Thursday', 'Friday', 'Saturday', 'Sunday'] * 3, rotation=45)
            plt.plot(x_plot[self.labels == cluster].T, c='slategrey', alpha=0.5)
            plt.plot(x_plot[self.labels == cluster].T.mean(axis=1), c='mediumvioletred')
            plt.grid()
            plt.title('cluster ' + str(cluster + 1) + ' (' + str(list(self.labels).count(cluster)) + ' samples)')
        if self.n_cluster % 2 != 0:
            plt.subplot(n_subplot, 2, self.n_cluster + 1)
        plt.xticks(x, ['Monday', 'Tuesday', 'Wednesday',
                       'Thursday', 'Friday', 'Saturday', 'Sunday'] * 3, rotation=45)
        plt.suptitle(title)

        return fig

    # plot cluster centers
    def plot_cluster_centers(self, x_plot=None):
        """
        Plot the cluster centers based on the clustering results.

        Parameters:
        -----------
        x_plot : array-like or None, optional (default=None)
            Data to be plotted. If None, the input data used for clustering will be plotted.

        Returns:
        --------
        Visualization of cluster centers.

        Notes:
        ------
        This method plots the cluster centers based on the clustering results. Each cluster center is represented
        by the mean of the data points within that cluster.
        """

        import matplotlib.pyplot as plt

        if x_plot is None:
            x_plot = self.x_input

        plt.figure(figsize=[18, 10])
        l = list()
        for i in range(self.n_cluster):
            # plt.plot(model.cluster_centers_[i])
            plt.plot(x_plot[self.labels == i].T.mean(axis=1))
            l.append('cluster ' + str(i + 1))
        plt.grid()
        plt.title('cluster centers')
        plt.legend(l)

    # plot every sample of specific cluster in single plot
    def plot_single_sample(self, cluster, n_sample=None, x_reference=None):
        """
        Plot individual samples from a specific cluster.

        Parameters:
        -----------
        cluster : int
            The cluster number for which samples will be plotted.
        n_sample : int or None, optional (default=None)
            Number of samples to plot. If None, the default is 10 samples.
        x_reference : DataFrame or None, optional (default=None)
            Reference data for plotting. If provided, samples will be selected from this reference data.

        Returns:
        --------
        Visualization of samples from specified cluster.

        Notes:
        ------
        This method plots individual samples from a specific cluster. It selects the specified number of samples
        from the cluster and plots them. If reference data is provided, samples will be selected from the reference data;
        otherwise, samples will be selected from the input data used for clustering.
        """

        import numpy as np
        import matplotlib.pyplot as plt

        if n_sample is None:
            n_sample = 10
            # n_sample = np.count_nonzero(self.labels == cluster - 1)  # show all samples (not recommended)
        count = 1
        plt.subplots(n_sample, 1, sharex=True, sharey=True)
        plt.suptitle("Cluster " + str(cluster) + ' (first' + str(n_sample) + ' samples)')
        if x_reference is not None:
            df = x_reference[self.labels == cluster - 1].transpose()
        else:
            df = self.x_input[self.labels == cluster - 1].transpose()

        for mac in df.columns[:n_sample]:
            plt.subplot(n_sample, 1, count)
            plt.plot(df[mac])
            count = count + 1
            plt.grid(which='both')
            plt.minorticks_on()

    # get statistical features of cluster center
    def get_param_centers(self, x_reference=None):
        """
        Retrieve statistical features of cluster centers.

        Parameters:
        -----------
        x_reference : DataFrame or None, optional (default=None)
            Reference data for extracting cluster centers. If provided, cluster centers will be computed based
            on this reference data; otherwise, cluster centers will be computed based on the input data used for clustering.

        Returns:
        --------
        param_cc : DataFrame
            Pandas DataFrame containing statistical features of cluster centers.

        Notes:
        ------
        This method retrieves statistical features of cluster centers. It computes the cluster centers either
        from the input data used for clustering or from the provided reference data.
        """

        import pandas as pd
        import syndatagenerators.data_analysis.feature_extraction as feature_extraction

        if self.labels is None:
            self.clustering()

        # store cluster centers in DataFrame
        dfc = pd.DataFrame()
        count = 0
        if x_reference is None:
            for cc in self.model.cluster_centers_:
                dfc[count] = pd.DataFrame(cc)
                count = count + 1
        else:
            for cluster in range(self.n_cluster):
                cc = x_reference[self.labels == cluster].T.mean(axis=1)
                dfc[count] = pd.DataFrame(cc)
                count = count + 1

        param_cc = feature_extraction.norm_param(feature_extraction.get_param(dfc))
        return param_cc

    # plot statistical feature of specific cluster center as heatmap
    def plot_heatmap_center(self, cc, x_reference=None):
        """
        Plot heatmap of statistical features for a specific cluster center.

        Parameters:
        -----------
        cc : int
            Index of the cluster center for which the heatmap will be plotted.
        x_reference : DataFrame or None, optional (default=None)
            Reference data for computing statistical features. If provided, statistical features will be
            computed based on this reference data; otherwise, statistical features will be computed based
            on the input data used for clustering.

        Returns:
        --------
        param_cc : DataFrame
            Statistical features of all cluster centers.

        Notes:
        ------
        This method plots a heatmap of statistical features for a specific cluster center. It computes the
        statistical features either from the input data used for clustering or from the provided reference data.
        """
        import syndatagenerators.data_analysis.feature_analysis as feature_analysis
        param_cc = self.get_param_centers(x_reference)
        feature_analysis.feature_heatmap(param_cc, cc)
        return param_cc[cc]

    # clustering validation with maximum mean discrepancy (mmd)
    def mmd_validation(self, df_raw, plot_hist=True):
        """
        Validate clustering using Maximum Mean Discrepancy (MMD) score.

        Parameters:
        -----------
        df_raw : DataFrame
            Raw data used for computing MMD score.
        plot_hist : bool, optional (default=True)
            Flag indicating whether to plot the histogram of MMD scores.

        Returns:
        --------
        mmd_sample : float
            MMD score for the clustering.

        Notes:
        ------
        This method validates clustering using the Maximum Mean Discrepancy (MMD) score, which measures the
        discrepancy between distributions of samples. It computes the MMD score based on the cluster labels
        and the provided raw data.
        """

        import syndatagenerators.data_analysis.mmd_score_clustering as mmd_score_clustering
        mmd_sample, mmd_sample_all, mmd_mean, mmd_mean_all = mmd_score_clustering.mmd_clustering_validation(
            self.labels, df_raw, plot_hist)
        return mmd_sample
