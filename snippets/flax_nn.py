"""
JAX/Flax Neural Network Reference for Jim Emulators

This file provides reference implementations for building the Speculator-style
neural network emulator in JAX/Flax. Based on:
- CosmoPower (2106.03846) - TensorFlow implementation
- Speculator (1911.11778) - Activation function (Eq. 4)

Framework: JAX + Flax (Linen API) + Optax (optimizers)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Sequence, Callable
from functools import partial

# Note: Don't set jax_enable_x64 in library modules (per GPT-5 review)
# Configuration should happen once at process start in the main script


# =============================================================================
# Speculator Activation Function
# =============================================================================

class SpeculatorActivation(nn.Module):
    """
    Speculator activation function from Alsing et al. (2019), Eq. 4.

    σ(x) = [γ + sigmoid(α·x) · (1 - γ)] · x

    Where α and γ are learnable parameters per neuron.
    γ is constrained to (0,1) via sigmoid on learnable logits.

    Properties:
    - When γ → 1: becomes linear (identity)
    - When γ → 0: becomes x·sigmoid(α·x) (smooth ReLU-like)
    - Learnable parameters allow network to interpolate between regimes
    - Smooth, infinitely differentiable (suitable for HMC)

    Updated based on GPT-5/Gemini review:
    - Uses shape inference (no explicit features arg)
    - Constrains γ ∈ (0,1) via sigmoid on logits
    """

    @nn.compact
    def __call__(self, x):
        # Infer feature dimension from input (lazy initialization)
        features = x.shape[-1]

        # α controls sigmoid steepness (unconstrained)
        alpha = self.param('alpha', nn.initializers.ones, (features,))

        # γ stored as logits, mapped through sigmoid to (0,1)
        # Initialize logits to -2.0 so sigmoid(-2) ≈ 0.12 (mostly gated)
        # Empirically: -2.0 outperforms +2.0 on toy task (1330x vs 176x improvement)
        gamma_logits = self.param('gamma_logits',
                                   nn.initializers.constant(-2.0),
                                   (features,))
        gamma = jax.nn.sigmoid(gamma_logits)

        # Speculator activation: (gamma + sigmoid(alpha * x) * (1 - gamma)) * x
        sigmoid_term = jax.nn.sigmoid(alpha * x)
        return (gamma + sigmoid_term * (1.0 - gamma)) * x


# Alternative: functional version for use in sequential models
def speculator_activation(x, alpha, gamma):
    """Functional Speculator activation."""
    sigmoid_term = jax.nn.sigmoid(alpha * x)
    return (gamma + sigmoid_term * (1.0 - gamma)) * x


# =============================================================================
# Emulator Network Architecture
# =============================================================================

class EmulatorMLP(nn.Module):
    """
    Multi-layer perceptron for waveform emulation.

    Architecture (from CosmoPower):
    - Input: normalized parameters θ (e.g., η, χ₁, χ₂)
    - Hidden layers: n_hidden layers with n_units each
    - Activation: Speculator activation (learnable α, γ per layer)
    - Output: PCA coefficients (linear, no activation)

    Typical config: 4 hidden layers × 512 units

    Updated based on GPT-5/Gemini review:
    - SpeculatorActivation uses shape inference (no features arg)
    """
    n_hidden: int = 4
    n_units: int = 512
    n_outputs: int = 50  # Number of PCA components

    @nn.compact
    def __call__(self, x):
        # Hidden layers with Speculator activation
        for i in range(self.n_hidden):
            x = nn.Dense(self.n_units, name=f'dense_{i}')(x)
            x = SpeculatorActivation(name=f'activation_{i}')(x)  # Shape inference

        # Output layer (linear, no activation)
        x = nn.Dense(self.n_outputs, name='output')(x)
        return x


class WaveformEmulator(nn.Module):
    """
    Full waveform emulator with built-in normalization.

    Handles:
    1. Input parameter normalization
    2. Network forward pass
    3. Output denormalization (PCA coefficients)

    Note: PCA reconstruction happens outside this module.
    """
    n_hidden: int = 4
    n_units: int = 512
    n_outputs: int = 50

    # These would be set from training data statistics
    params_mean: jnp.ndarray = None
    params_std: jnp.ndarray = None
    output_mean: jnp.ndarray = None
    output_std: jnp.ndarray = None

    @nn.compact
    def __call__(self, params):
        # Normalize inputs
        if self.params_mean is not None:
            x = (params - self.params_mean) / self.params_std
        else:
            x = params

        # Forward pass through MLP
        x = EmulatorMLP(
            n_hidden=self.n_hidden,
            n_units=self.n_units,
            n_outputs=self.n_outputs
        )(x)

        # Denormalize outputs
        if self.output_mean is not None:
            x = x * self.output_std + self.output_mean

        return x


# =============================================================================
# Training Utilities
# =============================================================================

def create_train_state(rng, model, learning_rate, input_shape):
    """
    Create initial training state.

    Args:
        rng: JAX random key
        model: Flax module
        learning_rate: Initial learning rate
        input_shape: Shape of input (e.g., (3,) for 3 parameters)

    Returns:
        TrainState with initialized parameters and optimizer
    """
    # Initialize parameters with dummy input
    params = model.init(rng, jnp.ones((1,) + input_shape))

    # AdamW optimizer with weight decay
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)

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
    min_lr: float = 1e-6
):
    """
    Create learning rate schedule with warmup and cosine decay.

    From CosmoPower training strategy:
    - Warmup: linear increase to init_lr
    - Decay: cosine annealing to min_lr
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=init_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=min_lr
    )
    return schedule


@jax.jit
def train_step(state, batch_params, batch_targets):
    """
    Single training step.

    Args:
        state: TrainState
        batch_params: Input parameters (batch_size, n_params)
        batch_targets: Target PCA coefficients (batch_size, n_pca)

    Returns:
        Updated state, loss value
    """
    def loss_fn(params):
        predictions = state.apply_fn(params, batch_params)
        # MSE loss on PCA coefficients
        return jnp.mean((predictions - batch_targets) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, batch_params, batch_targets):
    """Evaluation step (no gradient computation)."""
    predictions = state.apply_fn(state.params, batch_params)
    loss = jnp.mean((predictions - batch_targets) ** 2)
    return loss


# =============================================================================
# PCA Utilities (JAX-compatible)
# =============================================================================

class WaveformPCA:
    """
    PCA for waveform compression.

    Compresses high-dimensional waveforms (N_freq points) to
    low-dimensional representations (N_pca coefficients).

    Steps:
    1. Standardize: Y' = (Y - mean) / std
    2. SVD: Y' = U @ S @ V.T
    3. Keep top k components (99.99% variance)
    4. Coefficients: α = Y' @ V[:, :k]
    5. Reconstruction: Y' ≈ α @ V[:, :k].T
    """

    def __init__(self, n_components: int = None, explained_variance: float = 0.9999):
        self.n_components = n_components
        self.explained_variance = explained_variance
        self.mean_ = None
        self.std_ = None
        self.components_ = None  # Shape: (n_components, n_features)
        self.singular_values_ = None

    def fit(self, X):
        """
        Fit PCA on training data.

        Args:
            X: Training data, shape (n_samples, n_features)
        """
        # Standardize
        self.mean_ = jnp.mean(X, axis=0)
        self.std_ = jnp.std(X, axis=0)
        self.std_ = jnp.where(self.std_ == 0, 1.0, self.std_)  # Avoid division by zero

        X_std = (X - self.mean_) / self.std_

        # SVD (use numpy for fitting, JAX for inference)
        import numpy as np
        U, S, Vt = np.linalg.svd(np.array(X_std), full_matrices=False)

        # Determine number of components
        if self.n_components is None:
            # Keep enough for explained variance threshold
            explained_var_ratio = (S ** 2) / jnp.sum(S ** 2)
            cumsum = jnp.cumsum(explained_var_ratio)
            n_components = int(jnp.searchsorted(cumsum, self.explained_variance) + 1)
            self.n_components = min(n_components, len(S))

        self.components_ = jnp.array(Vt[:self.n_components])  # (n_components, n_features)
        self.singular_values_ = jnp.array(S[:self.n_components])

        return self

    def transform(self, X):
        """Project data to PCA space."""
        X_std = (X - self.mean_) / self.std_
        return X_std @ self.components_.T

    def inverse_transform(self, coeffs):
        """Reconstruct from PCA coefficients."""
        X_std = coeffs @ self.components_
        return X_std * self.std_ + self.mean_

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


# =============================================================================
# Complete Emulator Pipeline
# =============================================================================

class GWEmulator:
    """
    Complete gravitational wave emulator.

    Pipeline:
    1. Input: intrinsic parameters (η, χ₁, χ₂)
    2. Neural network: params → PCA coefficients
    3. PCA reconstruction: coefficients → log_amplitude, phase
    4. Output: amplitude, phase on geometric frequency grid

    Usage:
        emulator = GWEmulator.load('trained_model.pkl')
        amp, phase = emulator.predict(params)
    """

    def __init__(
        self,
        network: EmulatorMLP,
        pca_amplitude: WaveformPCA,
        pca_phase: WaveformPCA,
        frequency_grid: jnp.ndarray,
        params_mean: jnp.ndarray,
        params_std: jnp.ndarray,
    ):
        self.network = network
        self.pca_amplitude = pca_amplitude
        self.pca_phase = pca_phase
        self.frequency_grid = frequency_grid
        self.params_mean = params_mean
        self.params_std = params_std

    def predict(self, params, network_params):
        """
        Predict amplitude and phase for given parameters.

        Args:
            params: Intrinsic parameters (η, χ₁, χ₂), shape (n_samples, 3)
            network_params: Trained network parameters

        Returns:
            amplitude: Shape (n_samples, n_freq)
            phase: Shape (n_samples, n_freq)
        """
        # Normalize input parameters
        params_norm = (params - self.params_mean) / self.params_std

        # Network prediction (PCA coefficients for both amp and phase)
        pca_coeffs = self.network.apply(network_params, params_norm)

        # Split coefficients for amplitude and phase
        n_amp = self.pca_amplitude.n_components
        amp_coeffs = pca_coeffs[:, :n_amp]
        phase_coeffs = pca_coeffs[:, n_amp:]

        # PCA reconstruction
        log_amplitude = self.pca_amplitude.inverse_transform(amp_coeffs)
        phase = self.pca_phase.inverse_transform(phase_coeffs)

        # Convert log amplitude to amplitude
        amplitude = 10.0 ** log_amplitude

        return amplitude, phase


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Initialize and test network
    rng = jax.random.PRNGKey(0)

    # Create model
    model = EmulatorMLP(
        n_hidden=4,
        n_units=512,
        n_outputs=50  # Total PCA components (amp + phase)
    )

    # Initialize with dummy input (3 parameters: η, χ₁, χ₂)
    dummy_input = jnp.ones((1, 3))
    params = model.init(rng, dummy_input)

    # Test forward pass
    output = model.apply(params, dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")

    # Test Speculator activation (uses shape inference, no features arg)
    x = jnp.linspace(-3, 3, 100)
    act = SpeculatorActivation()
    act_params = act.init(rng, x.reshape(1, -1))
    y = act.apply(act_params, x.reshape(1, -1))
    print(f"\nSpeculator activation test:")
    print(f"  Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Output range: [{y.min():.2f}, {y.max():.2f}]")
