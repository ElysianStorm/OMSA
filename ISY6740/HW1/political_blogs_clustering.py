import numpy as np
from os.path import abspath, exists
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt # Although imported, plotting is not used in the final class method as per requirements.
from scipy.sparse import csgraph # Although csgraph is imported, it's not strictly used in this self-implemented version.

class PoliticalBlogsClustering:
    def __init__(self):
        # Base directory for data files. Assumes 'data' folder is in the same directory as the script.
        self.data_dir = "data/"
        self.nodes_file_path = abspath(self.data_dir + 'nodes.txt')
        self.edges_file_path = abspath(self.data_dir + 'edges.txt')

    def _load_data(self, nodes_file, edges_file):
        """
        Loads node and edge data from files.
        
        Args:
            nodes_file (str): Path to the nodes.txt file.
            edges_file (str): Path to the edges.txt file.
            
        Returns:
            tuple: A tuple containing:
                - nodes_df (pd.DataFrame): DataFrame of nodes with 'id' and 'label'.
                - edges_df (pd.DataFrame): DataFrame of edges with 'source' and 'target'.
        """
        # Load nodes data: id, blog_name, label, source_type
        nodes_df = pd.read_csv(nodes_file, sep='\t', header=None, names=['id', 'blog_name', 'label', 'source_type'])
        
        # Load edges data: source_id, target_id
        edges_df = pd.read_csv(edges_file, sep='\t', header=None, names=['source', 'target'])
        
        return nodes_df, edges_df

    def _preprocess_graph(self, nodes_df, edges_df):
        """
        Preprocesses the graph data: removes isolated nodes and creates adjacency matrix.
        
        Args:
            nodes_df (pd.DataFrame): DataFrame of nodes.
            edges_df (pd.DataFrame): DataFrame of edges.
            
        Returns:
            tuple: A tuple containing:
                - adj_matrix (np.ndarray): Symmetric adjacency matrix.
                - node_labels (np.ndarray): Labels of the remaining nodes, indexed by new_id.
                - id_to_new_id (dict): Mapping from original node ID to new contiguous ID.
                - new_id_to_id (dict): Mapping from new contiguous ID to original node ID.
        """
        # Get all unique nodes involved in edges
        connected_nodes = np.unique(np.concatenate((edges_df['source'].values, edges_df['target'].values)))
        
        # Filter nodes_df to include only connected nodes
        filtered_nodes_df = nodes_df[nodes_df['id'].isin(connected_nodes)].copy()
        
        # Create a mapping from original node IDs to new contiguous IDs (0 to N-1)
        # This is crucial because original IDs might not be contiguous or start from 0
        id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(filtered_nodes_df['id'].unique())}
        new_id_to_id = {new_id: old_id for old_id, new_id in id_to_new_id.items()}
        
        num_nodes = len(id_to_new_id)
        
        # Initialize adjacency matrix with zeros
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        
        # Populate adjacency matrix
        for _, row in edges_df.iterrows():
            u_orig, v_orig = row['source'], row['target']
            
            # Only add edge if both nodes exist in the filtered set
            if u_orig in id_to_new_id and v_orig in id_to_new_id:
                u_new = id_to_new_id[u_orig]
                v_new = id_to_new_id[v_orig]
                
                # Since the graph is undirected, set both (u,v) and (v,u) to 1
                adj_matrix[u_new, v_new] = 1
                adj_matrix[v_new, u_new] = 1
                
        # Get labels for the filtered nodes, ordered by their new_id
        # Ensure labels are retrieved in the order of new_id_to_id keys
        node_labels = filtered_nodes_df.set_index('id').loc[list(new_id_to_id.values())]['label'].values
        
        return adj_matrix, node_labels, id_to_new_id, new_id_to_id

    def _compute_laplacian(self, adj_matrix):
        """
        Computes the unnormalized Laplacian matrix L = D - A.
        
        Args:
            adj_matrix (np.ndarray): The adjacency matrix.
            
        Returns:
            np.ndarray: The unnormalized Laplacian matrix.
        """
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        laplacian_matrix = degree_matrix - adj_matrix
        return laplacian_matrix

    def _euclidean_distance(self, point1, point2):
        """Calculates the Euclidean distance between two points."""
        return np.sqrt(np.sum((point1 - point2)**2))

    def _kmeans(self, data, k, max_iterations=100, tolerance=1e-4):
        """
        Performs K-Means clustering.
        
        Args:
            data (np.ndarray): The data points to cluster.
            k (int): The number of clusters.
            max_iterations (int): Maximum number of iterations for K-Means.
            tolerance (float): Convergence tolerance for centroid movement.
            
        Returns:
            np.ndarray: An array of cluster assignments for each data point.
        """
        num_samples, num_features = data.shape
        
        # 1. Initialize centroids randomly
        # Select k random data points as initial centroids
        random_indices = np.random.choice(num_samples, k, replace=False)
        centroids = data[random_indices]
        
        for iteration in range(max_iterations):
            # 2. Assign each data point to the closest centroid
            cluster_assignments = np.zeros(num_samples, dtype=int)
            for i in range(num_samples):
                distances = [self._euclidean_distance(data[i], centroid) for centroid in centroids]
                cluster_assignments[i] = np.argmin(distances)
                
            # 3. Update centroids based on the mean of assigned points
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                points_in_cluster = data[cluster_assignments == j]
                if len(points_in_cluster) > 0:
                    new_centroids[j] = np.mean(points_in_cluster, axis=0)
                else:
                    # Handle empty cluster: re-initialize centroid to a random point
                    # This helps prevent issues if a cluster becomes empty
                    new_centroids[j] = data[np.random.choice(num_samples)]
            
            # 4. Check for convergence
            centroid_movement = np.sum([self._euclidean_distance(centroids[j], new_centroids[j]) for j in range(k)])
            if centroid_movement < tolerance:
                break
                
            centroids = new_centroids
            
        return cluster_assignments

    def _spectral_clustering(self, adj_matrix, k):
        """
        Performs spectral clustering on the given adjacency matrix.
        
        Args:
            adj_matrix (np.ndarray): The adjacency matrix.
            k (int): The number of clusters.
            
        Returns:
            np.ndarray: An array of cluster assignments for each node.
        """
        laplacian = self._compute_laplacian(adj_matrix)
        
        # Compute eigenvalues and eigenvectors
        # 'eigh' is used for symmetric matrices, which is more stable
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        # Sort eigenvalues and corresponding eigenvectors by eigenvalue magnitude
        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Select the k smallest non-zero eigenvectors
        # The first eigenvector (corresponding to eigenvalue 0) is trivial (all ones)
        # We look for eigenvalues slightly greater than zero due to numerical precision
        
        # Find indices of eigenvalues greater than a small tolerance
        non_zero_eigenvalue_indices = np.where(eigenvalues > 1e-8)[0]
        
        if len(non_zero_eigenvalue_indices) < k:
            # If not enough non-zero eigenvalues, use all available
            selected_eigenvector_indices = non_zero_eigenvalue_indices
        else:
            # Select the first k non-zero eigenvectors
            selected_eigenvector_indices = non_zero_eigenvalue_indices[:k]
        
        # Construct the feature matrix U from the selected eigenvectors
        U = eigenvectors[:, selected_eigenvector_indices]
        
        # Normalize rows of U to unit length (important for K-Means)
        row_sums = np.linalg.norm(U, axis=1)
        # Handle rows with zero norm (e.g., if a node has no connections and its eigenvector components are all zero)
        # Replace 0s in row_sums with 1s to avoid division by zero; these rows will remain 0 after division.
        row_sums[row_sums == 0] = 1 
        U_normalized = U / row_sums[:, np.newaxis]
        
        # Perform K-Means clustering on the rows of U_normalized
        cluster_assignments = self._kmeans(U_normalized, k)
        
        return cluster_assignments

    def _calculate_mismatch_rate(self, cluster_assignments, node_labels, k):
        """
        Calculates the mismatch rate for each cluster.
        
        Args:
            cluster_assignments (np.ndarray): Array of cluster assignments for each node.
            node_labels (np.ndarray): True labels of the nodes.
            k (int): Number of clusters.
            
        Returns:
            dict: A dictionary where keys are cluster IDs and values are their mismatch rates.
        """
        mismatch_rates = {}
        for cluster_id in range(k):
            # Get true labels for nodes in the current cluster
            labels_in_cluster = node_labels[cluster_assignments == cluster_id]
            
            if len(labels_in_cluster) == 0:
                mismatch_rates[cluster_id] = 0.0 # No mismatch if cluster is empty
                continue
                
            # Count occurrences of each label (0 and 1)
            counts = defaultdict(int)
            for label in labels_in_cluster:
                counts[label] += 1
                
            # Determine the majority label
            # If counts are equal, default to 0 as majority (arbitrary tie-break)
            if 0 in counts and 1 in counts:
                majority_label = 0 if counts[0] >= counts[1] else 1
            elif 0 in counts: # Only 0s present
                majority_label = 0
            else: # Only 1s present (or empty, handled above)
                majority_label = 1
                
            # Calculate mismatches
            mismatches = sum(1 for label in labels_in_cluster if label != majority_label)
            
            # Calculate mismatch rate
            mismatch_rate = mismatches / len(labels_in_cluster)
            mismatch_rates[cluster_id] = mismatch_rate
            
        return mismatch_rates

    def find_majority_labels(self, num_clusters = 2):
        '''
        This method loads the data, performs spectral clustering and reports the majority labels

        Inputs:
            num_clusters (int): The number of clusters to be created

        Output:
            A map with following attributes
            1. overall_mismatch_rate: <2 decimal places>
            2. mismatch_rates: [{"majority_index": <int>, "mismatch_rate": <2 decimal places>}]
        '''

        result_map = {
            "overall_mismatch_rate": None,
            "mismatch_rates": []
        }

        # Check if data files exist
        if not exists(self.nodes_file_path):
            print(f"Error: nodes.txt not found at {self.nodes_file_path}")
            return result_map
        if not exists(self.edges_file_path):
            print(f"Error: edges.txt not found at {self.edges_file_path}")
            return result_map

        # Load data
        nodes_df, edges_df = self._load_data(self.nodes_file_path, self.edges_file_path)
        
        # Preprocess graph (remove isolated nodes, create adjacency matrix)
        adj_matrix, node_labels, _, _ = self._preprocess_graph(nodes_df, edges_df)
        
        # Perform spectral clustering
        cluster_assignments = self._spectral_clustering(adj_matrix, num_clusters)
        
        # Calculate mismatch rates for each cluster
        mismatch_rates_dict = self._calculate_mismatch_rate(cluster_assignments, node_labels, num_clusters)
        
        # Prepare the "mismatch_rates" list for the output map
        formatted_mismatch_rates = []
        total_mismatches = 0
        total_nodes_in_clusters = 0

        for cluster_id, rate in mismatch_rates_dict.items():
            labels_in_cluster = node_labels[cluster_assignments == cluster_id]
            if len(labels_in_cluster) > 0:
                counts = defaultdict(int)
                for label in labels_in_cluster:
                    counts[label] += 1
                
                # Determine the majority label
                if 0 in counts and 1 in counts:
                    majority_label = 0 if counts[0] >= counts[1] else 1
                elif 0 in counts:
                    majority_label = 0
                else:
                    majority_label = 1
                
                mismatches_in_cluster = sum(1 for label in labels_in_cluster if label != majority_label)
                total_mismatches += mismatches_in_cluster
                total_nodes_in_clusters += len(labels_in_cluster)

                formatted_mismatch_rates.append({
                    "majority_index": cluster_id, # The cluster_id itself is the majority_index here
                    "mismatch_rate": round(rate, 2)
                })
            else:
                # Handle empty clusters as per the notebook's calculate_mismatch_rate
                formatted_mismatch_rates.append({
                    "majority_index": cluster_id,
                    "mismatch_rate": 0.00
                })
        
        # Calculate overall mismatch rate
        overall_mismatch_rate = 0.0
        if total_nodes_in_clusters > 0:
            overall_mismatch_rate = total_mismatches / total_nodes_in_clusters
        
        result_map["overall_mismatch_rate"] = round(overall_mismatch_rate, 2)
        result_map["mismatch_rates"] = formatted_mismatch_rates

        return result_map