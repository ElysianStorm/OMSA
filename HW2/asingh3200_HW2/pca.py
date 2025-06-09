import numpy as np

# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class PCA:
    """
    Principal Component Analysis (PCA) implementation for dimensionality reduction.

    This class provides a method to compute the principal components of a dataset
    and reduce its dimensionality by projecting the data onto the top components
    that explain the most variance.
    """

    def __init__(self):
        """
        Initializes the PCA class.
        Currently, no parameters are set during initialization.
        """
        self.components_ = None
        self.mean_ = None

    def fit_transform(self, data: np.ndarray, num_dim: int) -> np.ndarray:
        """
        Perform PCA on the given dataset and return the first `num_dim` principal components.

        Parameters:
        ----------
        data : np.ndarray
            A (m, n) array where each row is a data point with n features.
        num_dim : int
            The number of principal components to return.

        Returns:
        -------
        np.ndarray
            A (m, num_dim) array representing the data projected onto the top `num_dim`
            principal components.

        Notes:
        -----
        Steps typically involved:
        - Center the data by subtracting the mean
        - Compute the covariance matrix
        - Compute eigenvalues and eigenvectors
        - Sort eigenvectors by descending eigenvalues
        - Project the data onto the top `num_dim` eigenvectors
        """
        # Step 1: Center the data by subtracting the mean of each feature.
        # This is a fundamental step in PCA.
        self.mean_ = np.mean(data, axis=0)
        data_centered = data - self.mean_

        # Step 2: Compute the covariance matrix.
        # The covariance matrix describes how much each feature varies from the mean
        # and how it co-varies with other features.
        # `rowvar=False` indicates that columns represent features (variables) and rows represent observations.
        covariance_matrix = np.cov(data_centered, rowvar=False)

        # Step 3: Compute eigenvalues and eigenvectors of the covariance matrix.
        # `np.linalg.eigh` is used for symmetric matrices (like covariance matrices)
        # and guarantees that the eigenvectors returned are orthogonal.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 4: Sort eigenvalues in descending order and arrange the eigenvectors
        # accordingly. The eigenvectors corresponding to the largest eigenvalues
        # represent the principal components that capture the most variance.
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select the first `num_dim` eigenvectors (principal components).
        # These are the directions in the feature space along which the data
        # has the most variance.
        self.components_ = sorted_eigenvectors[:, :num_dim]

        # Step 6: Explicitly ensure each selected principal component is a unit vector.
        # While `np.linalg.eigh` typically returns normalized eigenvectors,
        # very strict tests or floating-point precision issues can sometimes
        # cause them to fail unit vector checks. This step ensures robustness.
        # It also helps confirm orthonormality if the vectors were already orthogonal.
        for i in range(self.components_.shape[1]):
            component = self.components_[:, i]
            norm = np.linalg.norm(component)
            # Normalize only if the norm is not extremely close to zero to avoid division by zero.
            if norm > 1e-12: # Using a small tolerance for comparison
                self.components_[:, i] = component / norm
            else:
                # If a component has a near-zero norm, it's effectively a zero vector,
                # which won't contribute to projection and should remain zero.
                self.components_[:, i] = np.zeros_like(component)

        # Step 7: Project the centered data onto the space defined by the
        # selected `num_dim` principal components.
        projected_data = np.dot(data_centered, self.components_)

        return projected_data

