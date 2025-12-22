#!/usr/bin/env python
"""
Toy training example for jim-emulators.

This script demonstrates the full training pipeline with synthetic data,
verifying that the JAX/Flax/Optax setup works correctly before using
real waveform data.

The toy problem: Learn a simple function mapping 3D input to 50D output
(mimicking: (eta, chi1, chi2) -> PCA coefficients)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from tqdm import tqdm

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Add src to path
import sys
sys.path.insert(0, "src")


# =============================================================================
# Import our network components
# =============================================================================

# Import from our snippet (or could be from jim_emulators.components later)
sys.path.insert(0, "snippets")
from flax_nn import SpeculatorActivation, EmulatorMLP, WaveformPCA


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_data(n_samples: int, n_outputs: int = 50, seed: int = 42):
    """
    Generate synthetic training data.

    Creates a smooth, nonlinear mapping from 3D input to n_outputs dimensions.
    This mimics the structure of (eta, chi1, chi2) -> PCA coefficients.
    """
    rng = np.random.default_rng(seed)

    # Input parameters (normalized to [0, 1])
    X = rng.uniform(0, 1, size=(n_samples, 3))

    # Create a smooth nonlinear target function
    # Each output is a combination of sinusoids and polynomials
    Y = np.zeros((n_samples, n_outputs))

    for i in range(n_outputs):
        # Different frequency/phase for each output dimension
        freq = 2 * np.pi * (i + 1) / n_outputs
        phase = i * 0.1

        # Combine inputs nonlinearly
        Y[:, i] = (
            np.sin(freq * X[:, 0] + phase) * np.cos(freq * X[:, 1]) +
            0.5 * X[:, 2] ** 2 +
            0.3 * X[:, 0] * X[:, 1] +
            0.1 * np.sin(3 * freq * X[:, 2])
        )

    # Add small noise
    Y += rng.normal(0, 0.01, Y.shape)

    return X.astype(np.float64), Y.astype(np.float64)


# =============================================================================
# Training Functions
# =============================================================================

def create_train_state(rng, model, learning_rate, input_dim):
    """Initialize training state."""
    params = model.init(rng, jnp.ones((1, input_dim)))
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@jax.jit
def train_step(state, batch_x, batch_y):
    """Single training step."""
    def loss_fn(params):
        predictions = state.apply_fn(params, batch_x)
        return jnp.mean((predictions - batch_y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_loss(state, x, y):
    """Evaluate loss on a dataset."""
    predictions = state.apply_fn(state.params, x)
    return jnp.mean((predictions - y) ** 2)


def train_epoch(state, train_x, train_y, batch_size, rng):
    """Train for one epoch."""
    n_samples = len(train_x)
    n_batches = n_samples // batch_size

    # Shuffle
    perm = jax.random.permutation(rng, n_samples)
    train_x = train_x[perm]
    train_y = train_y[perm]

    epoch_loss = 0.0
    for i in range(n_batches):
        batch_x = train_x[i * batch_size:(i + 1) * batch_size]
        batch_y = train_y[i * batch_size:(i + 1) * batch_size]
        state, loss = train_step(state, batch_x, batch_y)
        epoch_loss += loss

    return state, epoch_loss / n_batches


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    print("=" * 60)
    print("Toy Training Example for Jim Emulators")
    print("=" * 60)

    # Configuration
    n_train = 10000
    n_val = 1000
    n_outputs = 50
    n_hidden = 4
    n_units = 256  # Smaller for toy example
    batch_size = 256
    n_epochs = 100
    learning_rate = 1e-3

    print(f"\nConfiguration:")
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_val}")
    print(f"  Output dimensions: {n_outputs}")
    print(f"  Hidden layers: {n_hidden} x {n_units}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    train_x, train_y = generate_synthetic_data(n_train, n_outputs, seed=42)
    val_x, val_y = generate_synthetic_data(n_val, n_outputs, seed=123)

    # Convert to JAX arrays
    train_x = jnp.array(train_x)
    train_y = jnp.array(train_y)
    val_x = jnp.array(val_x)
    val_y = jnp.array(val_y)

    print(f"  Train X shape: {train_x.shape}")
    print(f"  Train Y shape: {train_y.shape}")

    # Create model
    print("\nInitializing model...")
    model = EmulatorMLP(
        n_hidden=n_hidden,
        n_units=n_units,
        n_outputs=n_outputs
    )

    # Initialize training state
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, learning_rate, input_dim=3)

    # Count parameters
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"  Total parameters: {n_params:,}")

    # Initial loss
    initial_train_loss = eval_loss(state, train_x, train_y)
    initial_val_loss = eval_loss(state, val_x, val_y)
    print(f"\nInitial losses:")
    print(f"  Train: {initial_train_loss:.6f}")
    print(f"  Val:   {initial_val_loss:.6f}")

    # Training loop
    print("\nTraining...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        rng, epoch_rng = jax.random.split(rng)
        state, train_loss = train_epoch(state, train_x, train_y, batch_size, epoch_rng)
        val_loss = eval_loss(state, val_x, val_y)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"  Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Final evaluation
    final_train_loss = eval_loss(state, train_x, train_y)
    final_val_loss = eval_loss(state, val_x, val_y)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal losses:")
    print(f"  Train: {final_train_loss:.6f} (initial: {initial_train_loss:.6f})")
    print(f"  Val:   {final_val_loss:.6f} (initial: {initial_val_loss:.6f})")
    print(f"  Improvement: {initial_val_loss / final_val_loss:.1f}x")

    # Test prediction
    print("\nTest prediction:")
    test_input = jnp.array([[0.5, 0.5, 0.5]])
    prediction = state.apply_fn(state.params, test_input)
    print(f"  Input: {test_input[0]}")
    print(f"  Output shape: {prediction.shape}")
    print(f"  Output range: [{prediction.min():.3f}, {prediction.max():.3f}]")

    # Verify gradients work (important for HMC)
    print("\nVerifying gradients...")
    def predict_fn(x):
        return state.apply_fn(state.params, x).sum()

    grad_fn = jax.grad(predict_fn)
    grads = grad_fn(test_input)
    print(f"  Gradient shape: {grads.shape}")
    print(f"  Gradient magnitude: {jnp.linalg.norm(grads):.6f}")
    print("  Gradients computed successfully!")

    print("\n" + "=" * 60)
    print("JAX/Flax training pipeline verified!")
    print("=" * 60)


if __name__ == "__main__":
    main()
