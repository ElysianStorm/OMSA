import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from os.path import abspath, exists

class MNISTClustering:
    def __init__(self, n_clusters=10, data_file='data/mnist_10digits.mat'):
        """
        Initializes the MNISTClustering class.

        Args:
            n_clusters (int): The number of clusters to form. Default is 10 for MNIST digits.
            data_file (str): The path to the .mat file containing MNIST data.
        """
        self.n_clusters = n_clusters
        self.data_file = abspath(data_file) # Ensure absolute path for data file

    def _compute_purity(self, labels, true_labels):
        """
        Calculates the purity score for each cluster.
        Purity is a measure of how well a clustering algorithm performs.
        For each cluster, it finds the most frequent true label and calculates
        the ratio of that count to the total number of data points in the cluster.

        Args:
            labels (np.ndarray): Predicted cluster assignments for each data point (output from KMeans).
            true_labels (np.ndarray): True labels of the data points.

        Returns:
            List of dictionaries with each cluster's purity score:
            [{"true_label": <label>, "purity_score": <purity_score>}]
            Returns 0.0 for purity if a cluster is empty.
        """
        purity_results = []
        for i in range(self.n_clusters):
            # Get indices of data points assigned to the current cluster
            indices = np.where(labels == i)[0]

            if len(indices) == 0:
                # If the cluster is empty, its purity is 0.
                purity_results.append({"true_label": None, "purity_score": 0.0})
                continue

            # Get the true labels for the data points in this cluster
            cluster_true_labels = true_labels[indices]

            # Find the most common true label in this cluster
            # Counter.most_common(1) returns a list of (element, count) tuples
            most_common_label, count = Counter(cluster_true_labels).most_common(1)[0]

            # Calculate purity for this cluster
            purity = count / len(indices)
            purity_results.append({"true_label": int(most_common_label), "purity_score": purity})
        return purity_results

    def _custom_kmeans(self, X, k, metric='euclidean', max_iter=100, random_state=42):
        """
        A custom K-Means implementation that supports different distance metrics
        using scipy.spatial.distance.cdist. This is used for Manhattan distance.

        Args:
            X (np.ndarray): The data points to cluster.
            k (int): The number of clusters.
            metric (str): The distance metric to use. 'euclidean' or 'cityblock' (for Manhattan).
            max_iter (int): Maximum number of iterations for the K-Means algorithm.
            random_state (int): Seed for random initialization of centroids to ensure reproducibility.

        Returns:
            np.ndarray: An array of cluster assignments for each data point.
        """
        np.random.seed(random_state) # Set seed for reproducibility

        num_samples, _ = X.shape

        # 1. Initialize centroids randomly by picking k random data points from X
        random_indices = np.random.choice(num_samples, k, replace=False)
        centroids = X[random_indices]

        for iteration in range(max_iter):
            # 2. Assign each data point to the closest centroid based on the specified metric
            distances = cdist(X, centroids, metric=metric)
            labels = np.argmin(distances, axis=1)

            # 3. Update centroids based on the mean of assigned points
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                points_in_cluster = X[labels == i]
                if points_in_cluster.shape[0] > 0:
                    # Calculate the mean of all points assigned to this cluster
                    new_centroids[i] = np.mean(points_in_cluster, axis=0)
                else:
                    # Handle empty cluster: re-initialize centroid to a random point
                    # This helps prevent issues if a cluster becomes empty during iterations.
                    new_centroids[i] = X[np.random.choice(num_samples)]

            # 4. Check for convergence: if centroids haven't moved significantly, stop
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return labels

    def purity_scores(self, norm_distance=2):
        '''
        This method loads the MNIST dataset, performs clustering using the specified distance metric,
        and calculates the purity score.

        Inputs:
            norm_distance (int): The distance metric to use for clustering (1 for Manhattan, 2 for Euclidean).

        Output:
            List of dictionaries with each cluster's purity score:
            [{"true_label": <label>, "purity_score": <purity_score>}]
        '''

        results = []

        # Check if the data file exists before attempting to load
        if not exists(self.data_file):
            print(f"Error: MNIST data file not found at {self.data_file}")
            return results

        # Load the MNIST dataset
        try:
            data = loadmat(self.data_file)
            xtrain = data['xtrain'] # Image data
            ytrain = data['ytrain'].flatten() # True labels
        except Exception as e:
            print(f"Error loading MNIST data from {self.data_file}: {e}")
            return results

        # Normalize the pixel values to be between 0 and 1, which is common practice
        # for clustering algorithms, especially those based on distance.
        x_normalized = xtrain / 255.0

        if norm_distance == 2: # Euclidean Distance
            # Use scikit-learn's KMeans for Euclidean distance as it is highly optimized
            # and robust for this common metric.
            kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans_model.fit_predict(x_normalized)

            # Calculate purity scores based on the clustering results
            results = self._compute_purity(cluster_labels, ytrain)

        elif norm_distance == 1: # Manhattan Distance (L1 norm, 'cityblock' in scipy)
            # For Manhattan distance, we use the custom KMeans implementation
            # which allows specifying the distance metric via `cdist`.
            cluster_labels = self._custom_kmeans(x_normalized, self.n_clusters, metric='cityblock', random_state=42)

            # Calculate purity scores based on the clustering results
            results = self._compute_purity(cluster_labels, ytrain)

        else:
            # Handle invalid norm_distance input
            print("Error: Invalid norm_distance. Use 1 for Manhattan or 2 for Euclidean.")

        return results

