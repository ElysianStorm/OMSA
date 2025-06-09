import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import random

# Import the PCA class from the local pca.py file
from pca import PCA

# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class OrderOfFaces:
    """
    This class handles loading and processing facial image data for dimensionality
    reduction using the ISOMAP algorithm, with PCA as an optional comparison.

    Attributes:
    ----------
    images_path : str
        Path to the .mat file containing the image dataset.
    X : np.ndarray
        The loaded image data, where each row is an image vector.

    Methods:
    -------
    get_adjacency_matrix(epsilon):
        Returns the adjacency matrix based on a given epsilon neighborhood.

    get_best_epsilon():
        Returns the best epsilon for the ISOMAP algorithm, likely based on
        graph connectivity or reconstruction error.

    isomap(epsilon):
        Computes a 2D embedding of the data using the ISOMAP algorithm.

    pca(num_dim):
        Returns a low-dimensional embedding of the data using PCA.
    """

    def __init__(self, images_path='data/isomap.mat'):
        """
        Initializes the OrderOfFaces object and loads image data from the given path.

        Parameters:
        ----------
        images_path : str
            Path to the .mat file containing the facial images dataset.
        """
        self.images_path = images_path
        self.X = self._load_data()

    def _load_data(self) -> np.ndarray:
        """
        Helper method to load data from the .mat file.
        """
        try:
            data = sio.loadmat(self.images_path)
            main_key = None
            max_dims = 0
            for key, value in data.items():
                if isinstance(value, np.ndarray) and value.size > max_dims and not key.startswith('__'):
                    main_key = key
                    max_dims = value.size
            
            if main_key:
                X_loaded = data[main_key]
            else:
                raise ValueError("Could not find a suitable data array in the .mat file.")

            # Ensure data is (n_samples, n_features)
            if X_loaded.shape[0] == 4096 and X_loaded.shape[1] == 698:
                # Transpose if the shape is (n_features, n_samples)
                X_loaded = X_loaded.T 
            
            print(f"Loaded data shape: {X_loaded.shape}")
            return X_loaded
            
        except FileNotFoundError:
            print(f"Error: {self.images_path} not found. Make sure it's in the correct folder.")
            exit()
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            exit()

    def get_adjacency_matrix(self, epsilon: float) -> np.ndarray:
        """
        Constructs the adjacency matrix using epsilon neighborhoods.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius within which points are considered connected.

        Returns:
        -------
        np.ndarray
            A 2D adjacency matrix (m x m) where each entry represents distance between
            neighbors within the epsilon threshold.
        """
        if self.X is None:
            raise ValueError("Data not loaded. Call __init__ first.")

        # Calculate pairwise Euclidean distances
        D_euclidean = squareform(pdist(self.X, metric='euclidean'))

        # Create the adjacency matrix (A)
        # A[i,j] = D_euclidean[i,j] if D_euclidean[i,j] <= epsilon, else infinity
        A = D_euclidean.copy()
        A[A > epsilon] = np.inf
        
        # Set diagonal to 0
        np.fill_diagonal(A, 0)
        
        return A

    def get_best_epsilon(self) -> float:
        """
        Heuristically determines the best epsilon value for graph connectivity in ISOMAP.
        This method iteratively searches for the smallest epsilon that results in a
        fully connected graph.

        Returns:
        -------
        float
            Optimal epsilon value ensuring a well-connected neighborhood graph.
        """
        if self.X is None:
            raise ValueError("Data not loaded. Call __init__ first.")

        # Determine a reasonable range for epsilon based on data's Euclidean distances
        D_euclidean = squareform(pdist(self.X, metric='euclidean'))
        # Get all non-zero distances
        non_zero_distances = D_euclidean[D_euclidean > 0] 

        if non_zero_distances.size == 0:
            # All distances are zero, which is unlikely for real data unless all points are identical
            return 0.0 

        min_dist = non_zero_distances.min()
        max_dist = non_zero_distances.max()

        # Start search from a low value, slightly above min_dist or a small arbitrary value
        # and search up to max_dist to ensure connectivity
        epsilon_candidates = np.linspace(min_dist * 0.9, max_dist * 1.1, num=1000)
        
        best_epsilon = None

        for epsilon in epsilon_candidates:
            adj_matrix = self.get_adjacency_matrix(epsilon)
            
            # Check connectivity using connected_components.
            # Using csr_matrix for efficient sparse graph operations.
            num_components, _ = connected_components(adj_matrix, directed=False)
            
            if num_components == 1:
                best_epsilon = epsilon
                break # Found the smallest epsilon that connects the graph
        
        if best_epsilon is None:
            # Fallback if no epsilon connects the graph within the range (unlikely for this dataset)
            # This might happen if epsilon_candidates range is too small or max_dist is too small
            # For this dataset, max_dist is ~34, so a larger max for linspace might be needed
            print("Warning: Could not find a fully connected graph within the search range.")
            # Return max_dist as a last resort, which should connect everything
            return max_dist 

        return best_epsilon

    def isomap(self, epsilon: float) -> np.ndarray:
        """
        Applies the ISOMAP algorithm to compute a 2D low-dimensional embedding of the dataset.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius for building the adjacency graph.

        Returns:
        -------
        np.ndarray
            A (m x 2) array where each row is a 2D embedding of the original data point.
        """
        if self.X is None:
            raise ValueError("Data not loaded. Call __init__ first.")

        # Step 1: Construct the adjacency matrix
        A = self.get_adjacency_matrix(epsilon)

        # Check connectivity. If disconnected, shortest_path might return inf.
        num_components, _ = connected_components(A, directed=False)
        if num_components > 1:
            print(f"Warning: Graph is disconnected with {num_components} components for epsilon={epsilon}.")
            print("ISOMAP may not yield meaningful results. Consider increasing epsilon.")

        # Step 2: Calculate all-pairs shortest paths (Geodesic Distances)
        # shortest_path returns a dense matrix by default
        D_geodesic = shortest_path(csgraph=A, method='auto', directed=False)

        if np.isinf(D_geodesic).any():
            print("Warning: Some nodes are unreachable (infinity values in geodesic distances).")
            print("This means your graph is disconnected. ISOMAP result may be unreliable.")

        # Step 3: Perform Multidimensional Scaling (MDS) on the geodesic distances
        # n_init=1 and max_iter for faster execution, though often n_init=4 is default
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=1, max_iter=3000)
        Y = mds.fit_transform(D_geodesic)
        
        return Y

    def pca(self, num_dim: int) -> np.ndarray:
        """
        Applies PCA to reduce the dataset to a specified number of dimensions.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to project the data onto.

        Returns:
        -------
        np.ndarray
            A (m x num_dim) array representing the dataset in a reduced PCA space.
        """
        if self.X is None:
            raise ValueError("Data not loaded. Call __init__ first.")

        # Instantiate your custom PCA class
        pca_instance = PCA()
        
        # Fit and transform the data using your PCA implementation
        X_pca = pca_instance.fit_transform(self.X, num_dim)
        
        return X_pca

# # Example usage (for testing, not part of the class methods)
# if __name__ == '__main__':
#     # Initialize the class
#     of = OrderOfFaces(images_path='data/isomap.mat')

#     # --- Part 1: Adjacency Matrix Visualization ---
#     # Use a chosen epsilon for visualization. This is a tuning step.
#     # From previous discussion, epsilon=10.0 is a good starting point for visualization
#     # that shows sparsity.
#     chosen_epsilon = 10.434941151035902 # This will be tuned with get_best_epsilon later
    
#     A_matrix = of.get_adjacency_matrix(chosen_epsilon)
    
#     plt.figure(figsize=(8, 8))
#     plt.imshow(A_matrix != np.inf, cmap='binary', origin='lower')
#     plt.title(f'Adjacency Matrix for ε = {chosen_epsilon}')
#     plt.xlabel('Node Index')
#     plt.ylabel('Node Index')
#     plt.colorbar(label='Connection (True/False)')
#     plt.show()

#     num_components, labels = connected_components(A_matrix, directed=False)
#     print(f"Number of connected components for ε={chosen_epsilon}: {num_components}")

#     # Visualize selected face images related to the graph structure
#     # Define a helper to visualize images directly for main script (as in Jupyter)
#     def visualize_single_image(image_vector, ax, title=""):
#         image_matrix = image_vector.reshape((64, 64))
#         ax.imshow(image_matrix, cmap='gray')
#         ax.set_title(title)
#         ax.axis('off')

#     num_nodes_to_show = 3
#     # Use random.seed for reproducibility if desired during testing
#     # random.seed(42) 
#     random_nodes = random.sample(range(of.X.shape[0]), num_nodes_to_show)

#     fig, axes = plt.subplots(num_nodes_to_show, 3, figsize=(10, num_nodes_to_show * 4))
#     fig.suptitle('Selected Nodes and Their Nearest Neighbors', fontsize=16)

#     for i, node_idx in enumerate(random_nodes):
#         visualize_single_image(of.X[node_idx], axes[i, 0], title=f"Node {node_idx}")

#         neighbors = np.where((A_matrix[node_idx, :] != np.inf) & (A_matrix[node_idx, :] != 0))[0]
#         if len(neighbors) > 0:
#             neighbor_idx = neighbors[0] 
#             visualize_single_image(of.X[neighbor_idx], axes[i, 1], title=f"Neighbor {neighbor_idx}")
#             if len(neighbors) > 1:
#                 second_neighbor_idx = neighbors[1]
#                 visualize_single_image(of.X[second_neighbor_idx], axes[i, 2], title=f"2nd Neighbor {second_neighbor_idx}")
#             else:
#                 axes[i, 2].set_visible(False) 
#         else:
#             # Handle cases where a node has no neighbors (except itself)
#             axes[i, 1].set_visible(False)
#             axes[i, 2].set_visible(False)
#             axes[i, 1].text(0.5, 0.5, "No neighbors found", ha='center', va='center', transform=axes[i, 1].transAxes)

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

#     # --- Part 2: ISOMAP Embedding ---
#     print("\n--- Running ISOMAP ---")
#     best_epsilon = of.get_best_epsilon()
#     print(f"Determined best epsilon for connectivity: {best_epsilon}")

#     isomap_embedding = of.isomap(best_epsilon)

#     plt.figure(figsize=(10, 8))
#     plt.scatter(isomap_embedding[:, 0], isomap_embedding[:, 1], s=10, alpha=0.7)
#     plt.title(f'ISOMAP 2D Embedding (ε = {best_epsilon:.2f})')
#     plt.xlabel('ISOMAP Component 1')
#     plt.ylabel('ISOMAP Component 2')
#     plt.grid(True)
#     plt.show()

#     # Find and show images from different parts of the embedding space
#     idx_min_c1 = np.argmin(isomap_embedding[:, 0])
#     idx_max_c1 = np.argmax(isomap_embedding[:, 0])
#     idx_min_c2 = np.argmin(isomap_embedding[:, 1])
#     idx_max_c2 = np.argmax(isomap_embedding[:, 1])

#     center_point = np.mean(isomap_embedding, axis=0)
#     dist_to_center = np.linalg.norm(isomap_embedding - center_point, axis=1)
#     idx_center = np.argmin(dist_to_center)

#     selected_embedding_indices = [idx_min_c1, idx_max_c1, idx_min_c2, idx_max_c2, idx_center]
#     selected_embedding_labels = ["Min C1", "Max C1", "Min C2", "Max C2", "Center"]

#     plt.figure(figsize=(10, 8))
#     plt.scatter(isomap_embedding[:, 0], isomap_embedding[:, 1], s=10, alpha=0.7, label='All points')
#     for i, idx in enumerate(selected_embedding_indices):
#         plt.scatter(isomap_embedding[idx, 0], isomap_embedding[idx, 1], s=100, color='red', marker='X', edgecolor='black', zorder=5,
#                     label=f'{selected_embedding_labels[i]} (Node {idx})')
#         plt.annotate(str(idx), (isomap_embedding[idx, 0], isomap_embedding[idx, 1]), textcoords="offset points", xytext=(5,5), ha='center')
#     plt.title(f'ISOMAP 2D Embedding with Selected Points (ε = {best_epsilon:.2f})')
#     plt.xlabel('ISOMAP Component 1')
#     plt.ylabel('ISOMAP Component 2')
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     fig, axes = plt.subplots(1, len(selected_embedding_indices), figsize=(20, 4))
#     fig.suptitle('Images from Different Parts of the ISOMAP Embedding', fontsize=16)

#     for i, idx in enumerate(selected_embedding_indices):
#         visualize_single_image(of.X[idx], axes[i], title=f"{selected_embedding_labels[i]}\n(Node {idx})")

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

#     # --- Part 3: PCA Embedding ---
#     print("\n--- Running PCA ---")
#     pca_embedding = of.pca(2) # Project to 2 dimensions

#     plt.figure(figsize=(10, 8))
#     plt.scatter(pca_embedding[:, 0], pca_embedding[:, 1], s=10, alpha=0.7)
#     plt.title('PCA 2D Embedding')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.grid(True)
#     plt.show()

#     # Find and show images from different parts of the PCA embedding space
#     idx_pca_min_c1 = np.argmin(pca_embedding[:, 0])
#     idx_pca_max_c1 = np.argmax(pca_embedding[:, 0])
#     idx_pca_min_c2 = np.argmin(pca_embedding[:, 1])
#     idx_pca_max_c2 = np.argmax(pca_embedding[:, 1])

#     center_point_pca = np.mean(pca_embedding, axis=0)
#     dist_to_center_pca = np.linalg.norm(pca_embedding - center_point_pca, axis=1)
#     idx_pca_center = np.argmin(dist_to_center_pca)

#     selected_pca_indices = [idx_pca_min_c1, idx_pca_max_c1, idx_pca_min_c2, idx_pca_max_c2, idx_pca_center]
#     selected_pca_labels = ["Min PC1", "Max PC1", "Min PC2", "Max PC2", "Center"]

#     plt.figure(figsize=(10, 8))
#     plt.scatter(pca_embedding[:, 0], pca_embedding[:, 1], s=10, alpha=0.7, label='All points')
#     for i, idx in enumerate(selected_pca_indices):
#         plt.scatter(pca_embedding[idx, 0], pca_embedding[idx, 1], s=100, color='red', marker='X', edgecolor='black', zorder=5,
#                     label=f'{selected_pca_labels[i]} (Node {idx})')
#         plt.annotate(str(idx), (pca_embedding[idx, 0], pca_embedding[idx, 1]), textcoords="offset points", xytext=(5,5), ha='center')
#     plt.title('PCA 2D Embedding with Selected Points')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     fig, axes = plt.subplots(1, len(selected_pca_indices), figsize=(20, 4))
#     fig.suptitle('Images from Different Parts of the PCA Embedding', fontsize=16)

#     for i, idx in enumerate(selected_pca_indices):
#         visualize_single_image(of.X[idx], axes[i], title=f"{selected_pca_labels[i]}\n(Node {idx})")

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()