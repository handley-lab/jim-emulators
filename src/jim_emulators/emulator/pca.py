"""
PCA compression for gravitational waveforms.

Compresses high-dimensional waveforms (N_freq ~ 2000 points) to
low-dimensional representations (N_pca ~ 50-100 coefficients).

IMPORTANT: This PCA class assumes the input data is ALREADY NORMALIZED
(zero mean at each frequency). It does NOT perform internal standardization.
The per-frequency normalization should be done externally using
PerFrequencyNormalizer before calling this class.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import jax.numpy as jnp


class WaveformPCA:
    """
    PCA for waveform compression (without internal standardization).

    This class performs pure SVD-based dimensionality reduction.
    It assumes the input data has already been normalized externally.

    Pipeline:
    1. SVD: X = U @ S @ V.T  (no standardization!)
    2. Keep top k components (99.99% variance)
    3. Coefficients: α = X @ V[:, :k]
    4. Reconstruction: X ≈ α @ V[:, :k].T

    Attributes
    ----------
    n_components : int
        Number of PCA components to keep.
    explained_variance_target : float
        Target explained variance ratio (default 0.9999).
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
        self.components_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "WaveformPCA":
        """
        Fit PCA on training data.

        IMPORTANT: X should already be normalized (zero mean at each feature).
        This class does NOT perform internal standardization.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Normalized training data (e.g., normalized log_amplitude or phase).

        Returns
        -------
        self : WaveformPCA
            Fitted PCA object.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # SVD directly on the (externally normalized) data
        # No internal standardization to avoid double normalization
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Compute explained variance ratio
        explained_var = (S ** 2) / (n_samples - 1)
        total_var = explained_var.sum()
        if total_var > 0:
            explained_variance_ratio = explained_var / total_var
        else:
            explained_variance_ratio = np.zeros_like(explained_var)

        # Determine number of components
        if self.n_components is None:
            cumsum = np.cumsum(explained_variance_ratio)
            n_components = int(np.searchsorted(cumsum, self.explained_variance_target) + 1)
            self.n_components = min(n_components, len(S))
        else:
            self.n_components = min(self.n_components, len(S))

        # Store top components
        self.components_ = Vt[:self.n_components]  # Shape: (n_components, n_features)
        self.singular_values_ = S[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data to PCA space.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Normalized data to transform.

        Returns
        -------
        coeffs : ndarray, shape (n_samples, n_components)
            PCA coefficients.
        """
        X = np.asarray(X)
        return X @ self.components_.T

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
            Reconstructed (normalized) data.
        """
        coeffs = np.asarray(coeffs)
        return coeffs @ self.components_

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
            Original normalized data.

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
        return X @ self.components_.T

    def inverse_transform_jax(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible inverse transform for use in JIT-compiled functions."""
        return coeffs @ self.components_

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for saving."""
        return {
            "n_components": self.n_components,
            "explained_variance_target": self.explained_variance_target,
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
