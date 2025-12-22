"""
Per-frequency normalization for gravitational waveforms.

Following the training pipeline documentation (implementation/training-pipeline.tex):
- Compute mean and std at each frequency bin
- Apply σ floor to prevent division by near-zero
- Normalize outputs to zero mean, unit variance at each frequency
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import jax.numpy as jnp


class PerFrequencyNormalizer:
    """
    Per-frequency standardization for waveform outputs.

    For each frequency bin k, computes:
        μ_k = mean(X[:, k])
        σ_k = max(std(X[:, k]), σ_floor)
        X_norm[:, k] = (X[:, k] - μ_k) / σ_k

    The σ floor prevents numerical instability at frequencies where
    waveforms vary little across the parameter space.

    Attributes
    ----------
    mean_ : ndarray, shape (n_freq,)
        Mean at each frequency bin.
    std_ : ndarray, shape (n_freq,)
        Standard deviation at each frequency bin (with floor applied).
    sigma_floor : float
        Minimum allowed standard deviation.
    """

    def __init__(self, sigma_floor: float = 1e-6):
        """
        Initialize normalizer.

        Parameters
        ----------
        sigma_floor : float
            Minimum allowed standard deviation (default 1e-6).
        """
        self.sigma_floor = sigma_floor
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PerFrequencyNormalizer":
        """
        Compute normalization statistics from training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_freq)
            Training waveforms (log_amplitude or phase).

        Returns
        -------
        self : PerFrequencyNormalizer
        """
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Apply floor to prevent division by near-zero
        self.std_ = np.maximum(self.std_, self.sigma_floor)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply normalization.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_freq)
            Waveforms to normalize.

        Returns
        -------
        X_norm : ndarray, shape (n_samples, n_freq)
            Normalized waveforms (zero mean, unit variance at each frequency).
        """
        X = np.asarray(X)
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Reverse normalization.

        Parameters
        ----------
        X_norm : ndarray, shape (n_samples, n_freq)
            Normalized waveforms.

        Returns
        -------
        X : ndarray, shape (n_samples, n_freq)
            Original-scale waveforms.
        """
        X_norm = np.asarray(X_norm)
        return X_norm * self.std_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    # JAX-compatible methods for inference
    def transform_jax(self, X: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible transform."""
        return (X - self.mean_) / self.std_

    def inverse_transform_jax(self, X_norm: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible inverse transform."""
        return X_norm * self.std_ + self.mean_

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sigma_floor": self.sigma_floor,
            "mean_": self.mean_,
            "std_": self.std_,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerFrequencyNormalizer":
        """Deserialize from dictionary."""
        norm = cls(sigma_floor=d["sigma_floor"])
        norm.mean_ = d["mean_"]
        norm.std_ = d["std_"]
        return norm

    def __repr__(self) -> str:
        if self.mean_ is None:
            return "PerFrequencyNormalizer(not fitted)"
        return f"PerFrequencyNormalizer(n_freq={len(self.mean_)}, σ_floor={self.sigma_floor})"


class CoefficientStandardizer:
    """
    Standardization for PCA coefficients.

    Following CosmoPower, the PCA coefficients are standardized to
    zero mean and unit variance before being used as training targets.

    This ensures all coefficients have comparable scale, improving
    neural network training stability.

    Attributes
    ----------
    mean_ : ndarray, shape (n_components,)
        Mean of each coefficient.
    std_ : ndarray, shape (n_components,)
        Standard deviation of each coefficient.
    """

    def __init__(self, sigma_floor: float = 1e-10):
        """
        Initialize standardizer.

        Parameters
        ----------
        sigma_floor : float
            Minimum allowed standard deviation.
        """
        self.sigma_floor = sigma_floor
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, coeffs: np.ndarray) -> "CoefficientStandardizer":
        """
        Compute standardization statistics.

        Parameters
        ----------
        coeffs : ndarray, shape (n_samples, n_components)
            PCA coefficients from training data.

        Returns
        -------
        self : CoefficientStandardizer
        """
        coeffs = np.asarray(coeffs)
        self.mean_ = np.mean(coeffs, axis=0)
        self.std_ = np.std(coeffs, axis=0)
        self.std_ = np.maximum(self.std_, self.sigma_floor)
        return self

    def transform(self, coeffs: np.ndarray) -> np.ndarray:
        """Standardize coefficients."""
        coeffs = np.asarray(coeffs)
        return (coeffs - self.mean_) / self.std_

    def inverse_transform(self, coeffs_std: np.ndarray) -> np.ndarray:
        """Reverse standardization."""
        coeffs_std = np.asarray(coeffs_std)
        return coeffs_std * self.std_ + self.mean_

    def fit_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(coeffs)
        return self.transform(coeffs)

    # JAX-compatible methods
    def transform_jax(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible transform."""
        return (coeffs - self.mean_) / self.std_

    def inverse_transform_jax(self, coeffs_std: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible inverse transform."""
        return coeffs_std * self.std_ + self.mean_

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sigma_floor": self.sigma_floor,
            "mean_": self.mean_,
            "std_": self.std_,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CoefficientStandardizer":
        """Deserialize from dictionary."""
        std = cls(sigma_floor=d["sigma_floor"])
        std.mean_ = d["mean_"]
        std.std_ = d["std_"]
        return std

    def __repr__(self) -> str:
        if self.mean_ is None:
            return "CoefficientStandardizer(not fitted)"
        return f"CoefficientStandardizer(n_components={len(self.mean_)})"


class InputNormalizer:
    """
    Input parameter normalization.

    Maps physical parameters to [-1, 1] range:
    - η ∈ [0.05, 0.25] → [-1, 1]
    - χ₁z, χ₂z ∈ [-0.99, 0.99] → [-1, 1]
    """

    def __init__(
        self,
        eta_bounds: Tuple[float, float] = (0.05, 0.25),
        chi_bounds: Tuple[float, float] = (-0.99, 0.99),
    ):
        """
        Initialize input normalizer.

        Parameters
        ----------
        eta_bounds : tuple
            (min, max) for symmetric mass ratio.
        chi_bounds : tuple
            (min, max) for spin parameters.
        """
        self.eta_bounds = eta_bounds
        self.chi_bounds = chi_bounds

        # Precompute scaling factors
        self.eta_min, self.eta_max = eta_bounds
        self.eta_scale = 2.0 / (self.eta_max - self.eta_min)
        self.eta_shift = -(self.eta_min + self.eta_max) / (self.eta_max - self.eta_min)

        self.chi_min, self.chi_max = chi_bounds
        self.chi_scale = 2.0 / (self.chi_max - self.chi_min)
        self.chi_shift = -(self.chi_min + self.chi_max) / (self.chi_max - self.chi_min)

    def transform(self, params: np.ndarray) -> np.ndarray:
        """
        Normalize input parameters.

        Parameters
        ----------
        params : ndarray, shape (n_samples, 3)
            Physical parameters [η, χ₁z, χ₂z].

        Returns
        -------
        params_norm : ndarray, shape (n_samples, 3)
            Normalized parameters in [-1, 1].
        """
        params = np.asarray(params)
        params_norm = np.zeros_like(params)
        # η normalization
        params_norm[:, 0] = params[:, 0] * self.eta_scale + self.eta_shift
        # χ₁z, χ₂z normalization
        params_norm[:, 1] = params[:, 1] * self.chi_scale + self.chi_shift
        params_norm[:, 2] = params[:, 2] * self.chi_scale + self.chi_shift
        return params_norm

    def inverse_transform(self, params_norm: np.ndarray) -> np.ndarray:
        """
        Reverse normalization.

        Parameters
        ----------
        params_norm : ndarray, shape (n_samples, 3)
            Normalized parameters.

        Returns
        -------
        params : ndarray, shape (n_samples, 3)
            Physical parameters.
        """
        params_norm = np.asarray(params_norm)
        params = np.zeros_like(params_norm)
        # η
        params[:, 0] = (params_norm[:, 0] - self.eta_shift) / self.eta_scale
        # χ₁z, χ₂z
        params[:, 1] = (params_norm[:, 1] - self.chi_shift) / self.chi_scale
        params[:, 2] = (params_norm[:, 2] - self.chi_shift) / self.chi_scale
        return params

    def transform_jax(self, params: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible transform."""
        params_norm = jnp.zeros_like(params)
        params_norm = params_norm.at[:, 0].set(params[:, 0] * self.eta_scale + self.eta_shift)
        params_norm = params_norm.at[:, 1].set(params[:, 1] * self.chi_scale + self.chi_shift)
        params_norm = params_norm.at[:, 2].set(params[:, 2] * self.chi_scale + self.chi_shift)
        return params_norm

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "eta_bounds": self.eta_bounds,
            "chi_bounds": self.chi_bounds,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InputNormalizer":
        """Deserialize from dictionary."""
        return cls(
            eta_bounds=tuple(d["eta_bounds"]),
            chi_bounds=tuple(d["chi_bounds"]),
        )
