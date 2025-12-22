"""
Complete gravitational waveform emulator.

Combines PCA compression with neural network for fast waveform generation.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

from .pca import WaveformPCA
from .network import EmulatorMLP


class GWEmulator:
    """
    Complete gravitational wave emulator.

    Pipeline:
    1. Input: intrinsic parameters (η, χ₁z, χ₂z)
    2. Normalize parameters
    3. Neural network: normalized params → PCA coefficients
    4. PCA reconstruction: coefficients → log_amplitude, phase
    5. Output: amplitude, phase on geometric frequency grid (Mf)

    Attributes
    ----------
    network : EmulatorMLP
        Neural network model.
    network_params : dict
        Trained network parameters.
    pca_amplitude : WaveformPCA
        PCA for amplitude compression.
    pca_phase : WaveformPCA
        PCA for phase compression.
    frequency_grid : ndarray
        Geometric frequency grid (Mf).
    params_mean : ndarray
        Mean of training parameters.
    params_std : ndarray
        Std of training parameters.
    """

    def __init__(
        self,
        network: EmulatorMLP,
        network_params: Dict,
        pca_amplitude: WaveformPCA,
        pca_phase: WaveformPCA,
        frequency_grid: np.ndarray,
        params_mean: np.ndarray,
        params_std: np.ndarray,
    ):
        self.network = network
        self.network_params = network_params
        self.pca_amplitude = pca_amplitude
        self.pca_phase = pca_phase
        self.frequency_grid = frequency_grid
        self.params_mean = jnp.array(params_mean)
        self.params_std = jnp.array(params_std)

        # Store dimensions
        self.n_amp_components = pca_amplitude.n_components
        self.n_phase_components = pca_phase.n_components

        # Create JIT-compiled predict function
        self._predict_jit = jax.jit(self._predict_impl)

    def _predict_impl(
        self,
        params: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Internal prediction implementation (JIT-compiled).

        Parameters
        ----------
        params : ndarray, shape (n_samples, 3)
            Intrinsic parameters (η, χ₁z, χ₂z).

        Returns
        -------
        log_amplitude : ndarray, shape (n_samples, n_freq)
        phase : ndarray, shape (n_samples, n_freq)
        """
        # Normalize input parameters
        params_norm = (params - self.params_mean) / self.params_std

        # Network prediction (PCA coefficients)
        pca_coeffs = self.network.apply(self.network_params, params_norm)

        # Split coefficients for amplitude and phase
        amp_coeffs = pca_coeffs[:, :self.n_amp_components]
        phase_coeffs = pca_coeffs[:, self.n_amp_components:]

        # PCA reconstruction (using JAX-compatible methods)
        log_amplitude = self.pca_amplitude.inverse_transform_jax(amp_coeffs)
        phase = self.pca_phase.inverse_transform_jax(phase_coeffs)

        return log_amplitude, phase

    def predict(
        self,
        params: np.ndarray,
        return_log_amplitude: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict waveform amplitude and phase.

        Parameters
        ----------
        params : ndarray, shape (n_samples, 3) or (3,)
            Intrinsic parameters [η, χ₁z, χ₂z].
        return_log_amplitude : bool
            If True, return log10(amplitude). If False, return amplitude.

        Returns
        -------
        amplitude : ndarray, shape (n_samples, n_freq)
            Log10 amplitude (or linear amplitude if return_log_amplitude=False).
        phase : ndarray, shape (n_samples, n_freq)
            Unwrapped phase in radians.
        """
        params = np.atleast_2d(params)
        params = jnp.array(params)

        log_amp, phase = self._predict_jit(params)

        # Convert to numpy
        log_amp = np.array(log_amp)
        phase = np.array(phase)

        if return_log_amplitude:
            return log_amp, phase
        else:
            return 10.0 ** log_amp, phase

    def predict_strain(
        self,
        params: np.ndarray,
    ) -> np.ndarray:
        """
        Predict complex strain h(Mf).

        Parameters
        ----------
        params : ndarray, shape (n_samples, 3) or (3,)
            Intrinsic parameters [η, χ₁z, χ₂z].

        Returns
        -------
        strain : ndarray, shape (n_samples, n_freq)
            Complex strain h = A * exp(-i * phi).
        """
        amplitude, phase = self.predict(params, return_log_amplitude=False)
        return amplitude * np.exp(-1j * phase)

    # =========================================================================
    # Serialization
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Save emulator to file.

        Parameters
        ----------
        path : str
            Output path (will add .pkl extension if not present).
        """
        path = Path(path)
        if path.suffix != ".pkl":
            path = path.with_suffix(".pkl")

        data = {
            "network_config": {
                "n_hidden": self.network.n_hidden,
                "n_units": self.network.n_units,
                "n_outputs": self.network.n_outputs,
            },
            "network_params": self.network_params,
            "pca_amplitude": self.pca_amplitude.to_dict(),
            "pca_phase": self.pca_phase.to_dict(),
            "frequency_grid": np.array(self.frequency_grid),
            "params_mean": np.array(self.params_mean),
            "params_std": np.array(self.params_std),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "GWEmulator":
        """
        Load emulator from file.

        Parameters
        ----------
        path : str
            Path to saved emulator.

        Returns
        -------
        emulator : GWEmulator
            Loaded emulator.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Reconstruct network
        network = EmulatorMLP(**data["network_config"])

        # Reconstruct PCA objects
        pca_amplitude = WaveformPCA.from_dict(data["pca_amplitude"])
        pca_phase = WaveformPCA.from_dict(data["pca_phase"])

        return cls(
            network=network,
            network_params=data["network_params"],
            pca_amplitude=pca_amplitude,
            pca_phase=pca_phase,
            frequency_grid=data["frequency_grid"],
            params_mean=data["params_mean"],
            params_std=data["params_std"],
        )


# =============================================================================
# Training Utilities
# =============================================================================

def create_train_state(
    rng: jax.random.PRNGKey,
    model: EmulatorMLP,
    learning_rate: float,
    input_dim: int = 3,
) -> train_state.TrainState:
    """
    Create initial training state with gradient clipping.

    Parameters
    ----------
    rng : PRNGKey
        Random key for initialization.
    model : EmulatorMLP
        Network model.
    learning_rate : float
        Initial learning rate.
    input_dim : int
        Input dimension (default 3 for η, χ₁z, χ₂z).

    Returns
    -------
    state : TrainState
        Initial training state.
    """
    params = model.init(rng, jnp.ones((1, input_dim)))

    # Optimizer with gradient clipping for stability
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


def create_learning_rate_schedule(
    init_lr: float = 1e-3,
    warmup_epochs: int = 10,
    total_epochs: int = 500,
    steps_per_epoch: int = 100,
    min_lr: float = 1e-6,
) -> optax.Schedule:
    """
    Create learning rate schedule with warmup and cosine decay.

    Parameters
    ----------
    init_lr : float
        Peak learning rate.
    warmup_epochs : int
        Epochs for linear warmup.
    total_epochs : int
        Total training epochs.
    steps_per_epoch : int
        Steps per epoch.
    min_lr : float
        Minimum learning rate.

    Returns
    -------
    schedule : optax.Schedule
        Learning rate schedule.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=init_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=min_lr,
    )


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch_params: jnp.ndarray,
    batch_targets: jnp.ndarray,
) -> Tuple[train_state.TrainState, float]:
    """
    Single training step.

    Parameters
    ----------
    state : TrainState
        Current training state.
    batch_params : ndarray, shape (batch_size, n_params)
        Input parameters.
    batch_targets : ndarray, shape (batch_size, n_pca)
        Target PCA coefficients.

    Returns
    -------
    state : TrainState
        Updated training state.
    loss : float
        MSE loss value.
    """
    def loss_fn(params):
        predictions = state.apply_fn(params, batch_params)
        return jnp.mean((predictions - batch_targets) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_loss(
    state: train_state.TrainState,
    params: jnp.ndarray,
    targets: jnp.ndarray,
) -> float:
    """Evaluate MSE loss on a dataset."""
    predictions = state.apply_fn(state.params, params)
    return jnp.mean((predictions - targets) ** 2)
