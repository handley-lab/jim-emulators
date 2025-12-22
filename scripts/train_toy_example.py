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
import matplotlib.pyplot as plt
from pathlib import Path
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
    """Initialize training state with gradient clipping (per GPT-5/Gemini review)."""
    params = model.init(rng, jnp.ones((1, input_dim)))
    # Add gradient clipping for stability (recommended by both reviewers)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    )
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

    # Track losses for plotting
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        rng, epoch_rng = jax.random.split(rng)
        state, train_loss = train_epoch(state, train_x, train_y, batch_size, epoch_rng)
        val_loss = eval_loss(state, val_x, val_y)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

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

    # Create visualization
    print("\nGenerating plots...")
    output_dir = Path("figures/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Training curves
    ax = axes[0, 0]
    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, label='Train', linewidth=2)
    ax.semilogy(epochs, val_losses, label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Predictions vs Targets (first 5 output dimensions)
    ax = axes[0, 1]
    predictions = np.array(state.apply_fn(state.params, val_x))
    targets = np.array(val_y)
    for i in range(5):
        ax.scatter(targets[:100, i], predictions[:100, i], alpha=0.5, s=10, label=f'Dim {i}')
    # Perfect prediction line
    lims = [min(targets[:100, :5].min(), predictions[:100, :5].min()),
            max(targets[:100, :5].max(), predictions[:100, :5].max())]
    ax.plot(lims, lims, 'k--', linewidth=1, label='Perfect')
    ax.set_xlabel('Target')
    ax.set_ylabel('Prediction')
    ax.set_title('Predictions vs Targets (first 5 dims)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Residuals distribution
    ax = axes[1, 0]
    residuals = (predictions - targets).flatten()
    ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (Prediction - Target)')
    ax.set_ylabel('Density')
    ax.set_title(f'Residual Distribution (std={residuals.std():.4f})')
    ax.grid(True, alpha=0.3)

    # Plot 4: Per-dimension error
    ax = axes[1, 1]
    per_dim_mse = np.mean((predictions - targets) ** 2, axis=0)
    ax.bar(range(n_outputs), per_dim_mse, alpha=0.7)
    ax.set_xlabel('Output Dimension')
    ax.set_ylabel('MSE')
    ax.set_title('Per-Dimension MSE')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = output_dir / "toy_training_results.png"
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")

    # Plot 2: Ground truth vs emulation for individual samples
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Ground Truth vs Emulation: Example Predictions', fontsize=14)

    # Pick 6 random validation samples
    rng_plot = np.random.default_rng(42)
    sample_indices = rng_plot.choice(len(val_x), 6, replace=False)

    for idx, (ax, sample_idx) in enumerate(zip(axes.flat, sample_indices)):
        # Get input parameters for this sample
        params = val_x[sample_idx]
        true_output = np.array(val_y[sample_idx])
        pred_output = np.array(state.apply_fn(state.params, val_x[sample_idx:sample_idx+1]))[0]

        # Plot as "spectrum" (output dimension on x-axis)
        dims = np.arange(n_outputs)
        ax.plot(dims, true_output, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(dims, pred_output, 'r--', linewidth=2, label='Emulation', alpha=0.8)

        # Show input parameters in title
        ax.set_title(f'Sample {sample_idx}\n(x₀={params[0]:.2f}, x₁={params[1]:.2f}, x₂={params[2]:.2f})',
                    fontsize=10)
        ax.set_xlabel('Output Dimension')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    filepath2 = output_dir / "toy_predictions_overlay.png"
    plt.savefig(filepath2, dpi=150)
    plt.close()
    print(f"  Saved: {filepath2}")

    # Plot 3: How the output "spectrum" varies with one input parameter
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Output Spectrum vs Input Parameters (Truth vs Emulation)', fontsize=14)

    param_names = ['x₀ (like η)', 'x₁ (like χ₁)', 'x₂ (like χ₂)']
    cmap = plt.cm.viridis

    for param_idx, (ax, pname) in enumerate(zip(axes, param_names)):
        # Vary one parameter while fixing others at 0.5
        n_curves = 10
        param_vals = np.linspace(0.1, 0.9, n_curves)

        for i, pval in enumerate(param_vals):
            color = cmap(i / (n_curves - 1))

            # Create input with varied parameter
            test_input = np.array([[0.5, 0.5, 0.5]])
            test_input[0, param_idx] = pval

            # Get ground truth
            true_y = np.zeros(n_outputs)
            for j in range(n_outputs):
                freq = 2 * np.pi * (j + 1) / n_outputs
                phase = j * 0.1
                x = test_input[0]
                true_y[j] = (
                    np.sin(freq * x[0] + phase) * np.cos(freq * x[1]) +
                    0.5 * x[2] ** 2 +
                    0.3 * x[0] * x[1] +
                    0.1 * np.sin(3 * freq * x[2])
                )

            # Get prediction
            pred_y = np.array(state.apply_fn(state.params, jnp.array(test_input)))[0]

            # Plot truth (solid) and prediction (dashed)
            ax.plot(range(n_outputs), true_y, '-', color=color, linewidth=1.5, alpha=0.7)
            ax.plot(range(n_outputs), pred_y, '--', color=color, linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Output Dimension')
        ax.set_ylabel('Value')
        ax.set_title(f'Varying {pname}\n(solid=truth, dashed=emulation)')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.1, 0.9))
        sm.set_array([])

    # Single colorbar for all
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Parameter Value')

    plt.tight_layout()
    filepath3 = output_dir / "toy_parameter_variation.png"
    plt.savefig(filepath3, dpi=150)
    plt.close()
    print(f"  Saved: {filepath3}")

    print("\nDone!")


if __name__ == "__main__":
    main()
