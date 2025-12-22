"""
PCA compression for gravitational waveforms.

Compresses high-dimensional waveforms (N_freq ~ 1000 points) to
low-dimensional representations (N_pca ~ 20-60 coefficients).
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import jax.numpy as jnp


class WaveformPCA:
    """
    PCA for waveform compression.

    Compresses waveforms to PCA coefficients for neural network training.
    Uses standardization before SVD for numerical stability.

    Pipeline:
    1. Standardize: Y' = (Y - mean) / std
    2. SVD: Y' = U @ S @ V.T
    3. Keep top k components (99.99% variance)
    4. Coefficients: α = Y' @ V[:, :k]
    5. Reconstruction: Y' ≈ α @ V[:, :k].T

    Attributes
    ----------
    n_components : int
        Number of PCA components to keep.
    explained_variance_target : float
        Target explained variance ratio (default 0.9999).
    mean_ : ndarray
        Mean of training data, shape (n_features,).
    std_ : ndarray
        Std of training data, shape (n_features,).
    components_ : ndarray
        PCA basis vectors, shape (n_components, n_features).
    explained_variance_ratio_ : ndarray
        Explained variance per component.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        explained_variance_target: float = 0.9999,
    ):
        """
        Initialize PCA.

        Parameters
        ----------
        n_components : int, optional
            Fixed number of components. If None, determined by explained_variance_target.
        explained_variance_target : float
            Target cumulative explained variance (default 0.9999 = 99.99%).
        """
        self.n_components = n_components
        self.explained_variance_target = explained_variance_target

        # Fitted attributes
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "WaveformPCA":
        """
        Fit PCA on training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data (e.g., log_amplitude or phase).

        Returns
        -------
        self : WaveformPCA
            Fitted PCA object.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Standardize
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.std_ = np.where(self.std_ < 1e-10, 1.0, self.std_)

        X_std = (X - self.mean_) / self.std_

        # SVD (full matrices=False for efficiency)
        U, S, Vt = np.linalg.svd(X_std, full_matrices=False)

        # Compute explained variance ratio
        explained_var = (S ** 2) / (n_samples - 1)
        total_var = explained_var.sum()
        self.explained_variance_ratio_ = explained_var / total_var

        # Determine number of components
        if self.n_components is None:
            cumsum = np.cumsum(self.explained_variance_ratio_)
            n_components = int(np.searchsorted(cumsum, self.explained_variance_target) + 1)
            self.n_components = min(n_components, len(S))
        else:
            self.n_components = min(self.n_components, len(S))

        # Store top components
        self.components_ = Vt[:self.n_components]  # Shape: (n_components, n_features)
        self.singular_values_ = S[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data to PCA space.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        coeffs : ndarray, shape (n_samples, n_components)
            PCA coefficients.
        """
        X = np.asarray(X)
        X_std = (X - self.mean_) / self.std_
        return X_std @ self.components_.T

    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from PCA coefficients.

        Parameters
        ----------
        coeffs : ndarray, shape (n_samples, n_components)
            PCA coefficients.

        Returns
        -------
        X : ndarray, shape (n_samples, n_features)
            Reconstructed data.
        """
        coeffs = np.asarray(coeffs)
        X_std = coeffs @ self.components_
        return X_std * self.std_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def reconstruction_error(self, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute reconstruction error.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Original data.

        Returns
        -------
        mse : float
            Mean squared error of reconstruction.
        per_sample_mse : ndarray, shape (n_samples,)
            MSE per sample.
        """
        X = np.asarray(X)
        coeffs = self.transform(X)
        X_recon = self.inverse_transform(coeffs)

        per_sample_mse = np.mean((X - X_recon) ** 2, axis=1)
        mse = np.mean(per_sample_mse)

        return mse, per_sample_mse

    # =========================================================================
    # JAX-compatible methods for inference
    # =========================================================================

    def transform_jax(self, X: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible transform for use in JIT-compiled functions."""
        X_std = (X - self.mean_) / self.std_
        return X_std @ self.components_.T

    def inverse_transform_jax(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible inverse transform for use in JIT-compiled functions."""
        X_std = coeffs @ self.components_
        return X_std * self.std_ + self.mean_

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for saving."""
        return {
            "n_components": self.n_components,
            "explained_variance_target": self.explained_variance_target,
            "mean_": self.mean_,
            "std_": self.std_,
            "components_": self.components_,
            "singular_values_": self.singular_values_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WaveformPCA":
        """Deserialize from dictionary."""
        pca = cls(
            n_components=d["n_components"],
            explained_variance_target=d["explained_variance_target"],
        )
        pca.mean_ = d["mean_"]
        pca.std_ = d["std_"]
        pca.components_ = d["components_"]
        pca.singular_values_ = d["singular_values_"]
        pca.explained_variance_ratio_ = d["explained_variance_ratio_"]
        return pca

    def __repr__(self) -> str:
        if self.components_ is None:
            return "WaveformPCA(not fitted)"
        total_var = np.sum(self.explained_variance_ratio_)
        return (
            f"WaveformPCA(n_components={self.n_components}, "
            f"explained_variance={total_var:.4f})"
        )
