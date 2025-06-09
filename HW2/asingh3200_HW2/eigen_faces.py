import os
import numpy as np
import imageio
from sklearn.decomposition import PCA
from scipy.ndimage import zoom

# -----------------------------------------------------------------------------
# NOTE: This class should NOT be modified.
# -----------------------------------------------------------------------------
class EigenFacesResult:
    def __init__(
        self,
        subject_1_eigen_faces: np.ndarray,
        subject_2_eigen_faces: np.ndarray,
        s11: float,
        s12: float,
        s21: float,
        s22: float
    ):
        self.subject_1_eigen_faces = subject_1_eigen_faces
        self.subject_2_eigen_faces = subject_2_eigen_faces
        self.s11 = s11
        self.s12 = s12
        self.s21 = s21
        self.s22 = s22

# -----------------------------------------------------------------------------
# DO NOT change method signatures or return types
# -----------------------------------------------------------------------------
class EigenFaces:
    def __init__(self, images_root_directory="data/yalefaces"):
        self.root_dir = images_root_directory
        self.image_shape = (60, 80)  # Updated to match expected output shape
        self.num_components = 6

        self.subjects = {
            "subject01": {"train": [], "test": None},
            "subject02": {"train": [], "test": None}
        }

        self._load_images()

    def _downsample_and_flatten(self, image: np.ndarray) -> np.ndarray:
        zoom_factors = (
            self.image_shape[0] / image.shape[0],
            self.image_shape[1] / image.shape[1]
        )
        downsampled = zoom(image, zoom_factors)
        return downsampled.flatten()

    def _load_images(self):
        for filename in os.listdir(self.root_dir):
            filepath = os.path.join(self.root_dir, filename)
            if filename.startswith("subject01"):
                if "test" in filename:
                    self.subjects["subject01"]["test"] = self._downsample_and_flatten(imageio.imread(filepath))
                else:
                    self.subjects["subject01"]["train"].append(self._downsample_and_flatten(imageio.imread(filepath)))
            elif filename.startswith("subject02"):
                if "test" in filename:
                    self.subjects["subject02"]["test"] = self._downsample_and_flatten(imageio.imread(filepath))
                else:
                    self.subjects["subject02"]["train"].append(self._downsample_and_flatten(imageio.imread(filepath)))

        # Convert lists to arrays
        for subject in self.subjects:
            self.subjects[subject]["train"] = np.array(self.subjects[subject]["train"])

    def _compute_eigenfaces(self, data_matrix: np.ndarray):
        mean_face = np.mean(data_matrix, axis=0)
        centered_data = data_matrix - mean_face
        pca = PCA(n_components=self.num_components)
        pca.fit(centered_data)
        eigenfaces = pca.components_.reshape((self.num_components, *self.image_shape))
        return pca, eigenfaces, mean_face

    def _projection_residual(self, test_image: np.ndarray, pca: PCA, mean_face: np.ndarray) -> float:
        centered_test = test_image - mean_face
        projection = pca.inverse_transform(pca.transform([centered_test]))[0]
        residual = np.linalg.norm(test_image - projection) ** 2
        return residual

    def run(self) -> EigenFacesResult:
        # Compute eigenfaces and mean for Subject 1
        pca1, eigenfaces_1, mean1 = self._compute_eigenfaces(self.subjects["subject01"]["train"])

        # Compute eigenfaces and mean for Subject 2
        pca2, eigenfaces_2, mean2 = self._compute_eigenfaces(self.subjects["subject02"]["train"])

        # Get test images
        test1 = self.subjects["subject01"]["test"]
        test2 = self.subjects["subject02"]["test"]

        # Compute residuals
        s11 = self._projection_residual(test1, pca1, mean1)
        s12 = self._projection_residual(test1, pca2, mean2)
        s21 = self._projection_residual(test2, pca1, mean1)
        s22 = self._projection_residual(test2, pca2, mean2)

        return EigenFacesResult(
            subject_1_eigen_faces=eigenfaces_1,
            subject_2_eigen_faces=eigenfaces_2,
            s11=s11,
            s12=s12,
            s21=s21,
            s22=s22
        )
