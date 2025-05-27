from PIL import Image
import numpy as np
import time
import os

class KMeansImpl:
    def __init__(self, max_iterations=20, num_runs=1, random_seed=None):
        """
        Initializes KMeansImpl with parameters for the clustering algorithm.

        Parameters:
            max_iterations (int): Maximum number of iterations for K-means convergence.
            num_runs (int): Number of random initializations to try for each clustering task.
                            The run with the lowest distortion will be chosen.
            random_seed (int, optional): Seed for NumPy's random number generator to ensure
                                         reproducibility of centroid initialization.
        """
        self.max_iterations = max_iterations
        self.num_runs = num_runs
        self.random_seed = random_seed

    def load_image(self, image_name="data/flower.jpeg"):
        """
        Loads an image from the specified path and returns its NumPy array.
        It first checks if the image exists in a 'data/' subdirectory, then tries the direct path.

        Inputs:
            image_name (str, optional): Path to the image file. Defaults to "data/flower.jpeg".

        Outputs:
            np.ndarray: A 3D NumPy array representing the RGB image, shape (H, W, 3),
                        or None if the image cannot be loaded.
        """
        # Ensure path uses OS-specific separator
        if not os.path.isabs(image_name) and not image_name.startswith('data/'):
            # Assume it's a file name, try looking in 'data/'
            full_path = os.path.join("data", image_name)
        else:
            full_path = image_name # Use as is if it's already a full path or explicitly in data/

        if not os.path.exists(full_path):
            print(f"Error: Image not found at {full_path}. Please check the path and 'data/' directory.")
            return None
        
        try:
            # Open and convert to RGB to handle potential transparency or different modes consistently
            # This ensures the output is a 3D NumPy array (H, W, 3) as expected.
            return np.array(Image.open(full_path).convert('RGB'))
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            return None

    def _initialize_centroids(self, pixels_flat, k):
        """
        Randomly initializes k centroids from the pixel data.

        Parameters:
            pixels_flat (numpy.ndarray): Flattened image pixels (N, 3), scaled to [0, 1].
            k (int): Number of clusters.

        Returns:
            numpy.ndarray: Initial centroids (k, 3).
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        indices = np.random.choice(pixels_flat.shape[0], k, replace=False)
        return pixels_flat[indices]

    def _assign_clusters(self, pixels_flat, centroids, norm_distance):
        """
        Assigns each pixel to the closest centroid based on the specified distance metric.

        Parameters:
            pixels_flat (numpy.ndarray): Flattened image pixels (N, 3), scaled to [0, 1].
            centroids (numpy.ndarray): Current cluster centroids (k, 3), scaled to [0, 1].
            norm_distance (int): 1 for Manhattan (L1) distance or 2 for Euclidean (L2) distance.

        Returns:
            numpy.ndarray: An array of cluster assignments (labels) for each pixel (N,).
        """
        # Expand dimensions for broadcasting: pixels_flat (N, 1, 3), centroids (1, k, 3)
        diff = pixels_flat[:, np.newaxis, :] - centroids[np.newaxis, :, :]

        if norm_distance == 1: # Manhattan (L1) distance
            distances = np.sum(np.abs(diff), axis=2)
        elif norm_distance == 2: # Squared Euclidean (L2) distance
            # Using squared Euclidean for computational efficiency,
            # as argmin(sqrt(d)) is equivalent to argmin(d)
            distances = np.sum(diff**2, axis=2)
        else:
            raise ValueError("norm_distance must be 1 (Manhattan) or 2 (Euclidean).")

        return np.argmin(distances, axis=1)

    def _update_centroids(self, pixels_flat, assignments, k_actual):
        """
        Recalculates centroids as the mean of assigned pixels.
        Handles empty clusters by re-initializing them with a random data point from the dataset.

        Parameters:
            pixels_flat (numpy.ndarray): Flattened image pixels (N, 3), scaled to [0, 1].
            assignments (numpy.ndarray): Current cluster assignments (N,).
            k_actual (int): The number of clusters being used.

        Returns:
            numpy.ndarray: Updated centroids (k_actual, 3).
        """
        new_centroids = np.zeros((k_actual, pixels_flat.shape[1]))
        total_pixels = pixels_flat.shape[0]

        for j in range(k_actual):
            points_in_cluster = pixels_flat[assignments == j]
            if len(points_in_cluster) > 0:
                new_centroids[j] = np.mean(points_in_cluster, axis=0)
            else:
                # Re-initialize empty cluster centroid randomly from the original pixels
                if self.random_seed is not None:
                    np.random.seed(self.random_seed + j) # Vary seed for different empty clusters
                random_pixel_index = np.random.choice(total_pixels)
                new_centroids[j] = pixels_flat[random_pixel_index]
                # print(f"Warning: Cluster {j} became empty. Re-initializing centroid randomly.")
        return new_centroids

    def _calculate_distortion(self, pixels_flat, assignments, centroids, norm_distance):
        """
        Calculates the distortion (sum of squared distances for Euclidean, sum of absolute distances for Manhattan)
        for a given clustering.

        Parameters:
            pixels_flat (numpy.ndarray): Flattened image pixels (N, 3), scaled to [0, 1].
            assignments (numpy.ndarray): Cluster assignments for each pixel.
            centroids (numpy.ndarray): The final cluster centroids.
            norm_distance (int): The distance metric used (1 for Manhattan, 2 for Euclidean).

        Returns:
            float: The total distortion of the clustering.
        """
        distortion = 0.0
        for i in range(pixels_flat.shape[0]):
            assigned_centroid = centroids[assignments[i]]
            if norm_distance == 1: # Manhattan distance
                distortion += np.sum(np.abs(pixels_flat[i] - assigned_centroid))
            elif norm_distance == 2: # Squared Euclidean distance
                distortion += np.sum((pixels_flat[i] - assigned_centroid)**2)
        return distortion

    def compress(self, pixels, num_clusters, norm_distance=2):
        """
        Compresses the image using the K-Means clustering algorithm.
        It runs the K-Means algorithm multiple times with different random initializations
        and returns the best result based on the lowest distortion.

        Inputs:
            pixels (np.ndarray): A 3D array of shape (H, W, 3) containing RGB pixel data.
            num_clusters (int): Number of clusters for compression.
            norm_distance (int, optional): Distance metric. 1 = Manhattan, 2 = Euclidean (default).

        Outputs:
            dict: 
                "class" (np.ndarray, shape (H, W)): Cluster index for each pixel.
                "centroid" (np.ndarray, shape (num_clusters, 3)): RGB values of centroids.
                "img" (np.ndarray, shape (H, W, 3)): Compressed image as a NumPy array.
                "number_of_iterations" (int): Iterations taken by K-Means.
                "time_taken" (float): Time taken in seconds.
                "additional_args" (dict): Placeholder for any extra parameters.
        """
        # Store original image shape for reconstruction
        original_shape = pixels.shape
        # Flatten pixels to (N, 3) where N is total pixels, and normalize to [0, 1]
        pixels_flat = pixels.reshape(-1, 3) / 255.0

        best_assignments = None
        best_centroids = None
        best_num_iterations = 0
        best_time_taken = 0.0
        min_distortion = float('inf') # Initialize with a very large value

        for run_idx in range(self.num_runs):
            # Seed random for each run if a global seed is provided, or rely on system time
            if self.random_seed is not None:
                np.random.seed(self.random_seed + run_idx)

            start_time = time.time()
            
            # Initialize centroids for the current run
            centroids = self._initialize_centroids(pixels_flat, num_clusters)
            prev_assignments = None
            current_iterations = 0

            # K-Means iterative process
            for i in range(self.max_iterations):
                current_iterations = i + 1
                assignments = self._assign_clusters(pixels_flat, centroids, norm_distance)

                # Check for convergence: if assignments haven't changed from the previous iteration
                if prev_assignments is not None and np.array_equal(assignments, prev_assignments):
                    break # Converged, exit loop
                
                centroids = self._update_centroids(pixels_flat, assignments, num_clusters)
                prev_assignments = assignments # Store current assignments for next iteration's comparison
            
            time_taken = time.time() - start_time
            # Calculate distortion for the current run's final clustering
            current_distortion = self._calculate_distortion(pixels_flat, assignments, centroids, norm_distance)

            # Keep track of the best clustering found so far (lowest distortion)
            if current_distortion < min_distortion:
                min_distortion = current_distortion
                best_assignments = assignments
                best_centroids = centroids
                best_num_iterations = current_iterations
                best_time_taken = time_taken
        
        # Handle cases where no valid clustering was found (e.g., if num_runs was 0, or input was empty)
        if best_assignments is None or best_centroids is None:
            print("Error: K-Means did not converge or no valid clusters were formed across all runs.")
            print("This might happen if num_clusters is too high for the data, or input pixels are empty.")
            return {
                "class": None,
                "centroid": None,
                "img": None,
                "number_of_iterations": 0,
                "time_taken": 0.0,
                "distortion": float('inf'),
                "additional_args": {"error_message": "No valid clustering result obtained."}
            }

        # Reconstruct the image using the best centroids and assignments found
        # Scale centroids back to [0, 255] and convert to uint8
        reconstructed_pixels_normalized = best_centroids[best_assignments]
        reconstructed_image_array = (reconstructed_pixels_normalized * 255).astype(np.uint8)
        
        # Add a check for empty array after operations, before reshape
        if reconstructed_image_array.size == 0:
            print("Error: Reconstructed pixels array is empty. Cannot create image.")
            return {
                "class": best_assignments,
                "centroid": (best_centroids * 255).astype(np.uint8),
                "img": None, # Return None for image if it cannot be formed
                "number_of_iterations": best_num_iterations,
                "time_taken": best_time_taken,
                "distortion": min_distortion,
                "additional_args": {"error_message": "Reconstructed pixels array is empty."}
            }

        # Reshape back to original image dimensions (H, W, 3) as expected by the output
        # No longer using PIL.Image.fromarray for the output 'img'
        reconstructed_image_array = reconstructed_image_array.reshape(original_shape)

        # Prepare the result map as specified in the template
        result_map = {
            "class": best_assignments.reshape(original_shape[:-1]), # Reshape class to (H, W)
            "centroid": (best_centroids * 255).astype(np.uint8), # Centroids scaled back to 0-255
            "img": reconstructed_image_array, # Now a NumPy array (H, W, 3)
            "number_of_iterations": best_num_iterations,
            "time_taken": best_time_taken,
            "distortion": min_distortion,
            "additional_args": {} # Placeholder for any extra arguments
        }

        return result_map

