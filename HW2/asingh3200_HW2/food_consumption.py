from pca import PCA
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class FoodConsumptionPCA:
    """
    This class loads and processes a food consumption dataset for performing PCA
    from two perspectives:
    - Country-based PCA: Countries as samples, foods as features.
    - Food-based PCA: Foods as samples, country consumption patterns as features.
    """

    def __init__(self, input_path="data/food-consumption.csv"):
        """
        Initializes the FoodConsumptionPCA object and loads data from a CSV file.

        Parameters:
        ----------
        input_path : str
            Path to the CSV file containing the food consumption data.
        """
        try:
            self.df = pd.read_csv(input_path)
        except FileNotFoundError:
            print(f"Error: The file '{input_path}' was not found.")
            print("Please ensure 'food-consumption.csv' is in the specified path.")
            raise # Re-raise the error as the data is essential for the class to function.


    def country_pca(self, num_dim: int) -> np.ndarray:
        """
        Performs PCA where each row represents a country and each column represents a food item.

        This will reduce the feature space (foods) to `num_dim` principal components.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to retain.

        Returns:
        -------
        np.ndarray
            A (num_countries, num_dim) array representing countries in the reduced PCA space.
        """
        # Step 1: Prepare the data matrix for country PCA
        # Countries are observations (rows), food items are features (columns).
        # We set 'Country' column as index and convert to NumPy array.
        df_pca_countries = self.df.set_index('Country')
        data_for_pca = df_pca_countries.values

        # Step 2: Initialize and perform PCA
        # The PCA class handles mean-centering, covariance matrix calculation,
        # eigenvalue decomposition, and projection internally.
        pca_model = PCA()
        projected_data = pca_model.fit_transform(data_for_pca, num_dim)

        return projected_data

    def food_pca(self, num_dim: int) -> np.ndarray:
        """
        Performs PCA where each row represents a food item and each column represents a country.

        This will reduce the country-dimension feature space to `num_dim` principal components.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to retain.

        Returns:
        -------
        np.ndarray
            A (num_foods, num_dim) array representing foods in the reduced PCA space.
        """
        # Step 1: Prepare the data matrix for food item PCA
        # Food items are observations (rows), countries are features (columns).
        # We set 'Country' as index and then transpose the DataFrame.
        df_pca_food_items = self.df.set_index('Country').T
        data_for_pca = df_pca_food_items.values

        # Step 2: Initialize and perform PCA
        # The PCA class handles mean-centering, covariance matrix calculation,
        # eigenvalue decomposition, and projection internally.
        pca_model = PCA()
        projected_data = pca_model.fit_transform(data_for_pca, num_dim)

        return projected_data
