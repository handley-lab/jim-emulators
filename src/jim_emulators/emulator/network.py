"""
Neural network components for gravitational waveform emulation.

Based on:
- CosmoPower (2106.03846) - Architecture
- Speculator (1911.11778) - Activation function (Eq. 4)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


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

    This activation is critical for producing smooth gradients needed
    for Hamiltonian Monte Carlo inference.
    """

    @nn.compact
    def __call__(self, x):
        # Infer feature dimension from input (lazy initialization)
        features = x.shape[-1]

        # α controls sigmoid steepness (unconstrained)
        alpha = self.param('alpha', nn.initializers.ones, (features,))

        # γ stored as logits, mapped through sigmoid to (0,1)
        # Initialize logits to -2.0 so sigmoid(-2) ≈ 0.12 (mostly gated)
        # Empirically: -2.0 outperforms +2.0 (more nonlinear initially)
        gamma_logits = self.param(
            'gamma_logits',
            nn.initializers.constant(-2.0),
            (features,)
        )
        gamma = jax.nn.sigmoid(gamma_logits)

        # Speculator activation: (gamma + sigmoid(alpha * x) * (1 - gamma)) * x
        sigmoid_term = jax.nn.sigmoid(alpha * x)
        return (gamma + sigmoid_term * (1.0 - gamma)) * x


class EmulatorMLP(nn.Module):
    """
    Multi-layer perceptron for waveform emulation.

    Architecture (from CosmoPower):
    - Input: normalized parameters θ (e.g., η, χ₁, χ₂)
    - Hidden layers: n_hidden layers with n_units each
    - Activation: Speculator activation (learnable α, γ per layer)
    - Output: PCA coefficients (linear, no activation)

    Typical config: 4 hidden layers × 512 units

    Attributes
    ----------
    n_hidden : int
        Number of hidden layers (default 4).
    n_units : int
        Units per hidden layer (default 512).
    n_outputs : int
        Output dimension (total PCA components for amp + phase).
    """
    n_hidden: int = 4
    n_units: int = 512
    n_outputs: int = 50

    @nn.compact
    def __call__(self, x):
        # Hidden layers with Speculator activation
        for i in range(self.n_hidden):
            x = nn.Dense(self.n_units, name=f'dense_{i}')(x)
            x = SpeculatorActivation(name=f'activation_{i}')(x)

        # Output layer (linear, no activation)
        x = nn.Dense(self.n_outputs, name='output')(x)
        return x


class DualHeadMLP(nn.Module):
    """
    MLP with separate heads for amplitude and phase.

    This architecture has shared hidden layers but separate output
    layers for amplitude and phase PCA coefficients. This can help
    when amplitude and phase have different characteristics.

    Not used in the baseline, but available for experimentation.
    """
    n_hidden: int = 4
    n_units: int = 512
    n_amp_components: int = 30
    n_phase_components: int = 20

    @nn.compact
    def __call__(self, x):
        # Shared hidden layers
        for i in range(self.n_hidden):
            x = nn.Dense(self.n_units, name=f'shared_dense_{i}')(x)
            x = SpeculatorActivation(name=f'shared_activation_{i}')(x)

        # Separate output heads
        amp_out = nn.Dense(self.n_amp_components, name='amp_output')(x)
        phase_out = nn.Dense(self.n_phase_components, name='phase_output')(x)

        # Concatenate outputs
        return jnp.concatenate([amp_out, phase_out], axis=-1)
