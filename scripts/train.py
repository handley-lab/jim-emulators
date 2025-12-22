#!/usr/bin/env python
"""
Train the gravitational waveform emulator.

This script trains a neural network to predict PCA coefficients of
gravitational waveforms from intrinsic binary parameters.

Usage:
    python scripts/train.py                           # Default config
    python scripts/train.py --epochs 200 --batch_size 512
    python scripts/train.py --data data/waveforms_22mode.h5
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.emulator import WaveformPCA, EmulatorMLP, GWEmulator
from jim_emulators.emulator.emulator import create_train_state, train_step, eval_loss


# =============================================================================
# Data Loading
# =============================================================================

def load_training_data(path: str):
    """
    Load training data from HDF5 file.

    Returns
    -------
    data : dict
        Dictionary with train/val parameters, amplitude, phase, and frequency grid.
    """
    with h5py.File(path, "r") as f:
        data = {
            "train_params": f["train/parameters"][:],
            "train_log_amp": f["train/log_amplitude"][:],
            "train_phase": f["train/phase"][:],
            "val_params": f["validation/parameters"][:],
            "val_log_amp": f["validation/log_amplitude"][:],
            "val_phase": f["validation/phase"][:],
            "frequency_grid": f["frequency_grid"][:],
        }
    return data


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(state, train_params, train_targets, batch_size, rng):
    """Train for one epoch with shuffled batches."""
    n_samples = len(train_params)
    n_batches = n_samples // batch_size

    # Shuffle
    perm = jax.random.permutation(rng, n_samples)
    train_params = train_params[perm]
    train_targets = train_targets[perm]

    epoch_loss = 0.0
    for i in range(n_batches):
        batch_params = train_params[i * batch_size:(i + 1) * batch_size]
        batch_targets = train_targets[i * batch_size:(i + 1) * batch_size]
        state, loss = train_step(state, batch_params, batch_targets)
        epoch_loss += loss

    return state, epoch_loss / n_batches


def train_emulator(
    data: dict,
    n_hidden: int = 4,
    n_units: int = 512,
    batch_size: int = 256,
    n_epochs: int = 200,
    learning_rate: float = 1e-3,
    patience: int = 30,
    output_dir: str = "outputs",
):
    """
    Train the waveform emulator.

    Parameters
    ----------
    data : dict
        Training data from load_training_data.
    n_hidden : int
        Number of hidden layers.
    n_units : int
        Units per hidden layer.
    batch_size : int
        Training batch size.
    n_epochs : int
        Maximum epochs.
    learning_rate : float
        Initial learning rate.
    patience : int
        Early stopping patience.
    output_dir : str
        Output directory for model and plots.

    Returns
    -------
    emulator : GWEmulator
        Trained emulator.
    history : dict
        Training history.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Waveform Emulator")
    print("=" * 60)

    # ==========================================================================
    # Step 1: Fit PCA on training data
    # ==========================================================================
    print("\n[1/4] Fitting PCA on amplitude and phase...")

    pca_amplitude = WaveformPCA(explained_variance_target=0.9999)
    pca_phase = WaveformPCA(explained_variance_target=0.9999)

    pca_amplitude.fit(data["train_log_amp"])
    pca_phase.fit(data["train_phase"])

    print(f"  Amplitude PCA: {pca_amplitude.n_components} components "
          f"({np.sum(pca_amplitude.explained_variance_ratio_):.4f} variance)")
    print(f"  Phase PCA: {pca_phase.n_components} components "
          f"({np.sum(pca_phase.explained_variance_ratio_):.4f} variance)")

    # Check reconstruction error
    amp_mse, _ = pca_amplitude.reconstruction_error(data["train_log_amp"])
    phase_mse, _ = pca_phase.reconstruction_error(data["train_phase"])
    print(f"  Amplitude reconstruction MSE: {amp_mse:.2e}")
    print(f"  Phase reconstruction MSE: {phase_mse:.2e}")

    # ==========================================================================
    # Step 2: Prepare training data
    # ==========================================================================
    print("\n[2/4] Preparing training data...")

    # Transform waveforms to PCA coefficients
    train_amp_coeffs = pca_amplitude.transform(data["train_log_amp"])
    train_phase_coeffs = pca_phase.transform(data["train_phase"])
    train_targets = np.concatenate([train_amp_coeffs, train_phase_coeffs], axis=1)

    val_amp_coeffs = pca_amplitude.transform(data["val_log_amp"])
    val_phase_coeffs = pca_phase.transform(data["val_phase"])
    val_targets = np.concatenate([val_amp_coeffs, val_phase_coeffs], axis=1)

    # Normalize input parameters
    params_mean = np.mean(data["train_params"], axis=0)
    params_std = np.std(data["train_params"], axis=0)

    train_params_norm = (data["train_params"] - params_mean) / params_std
    val_params_norm = (data["val_params"] - params_mean) / params_std

    # Convert to JAX arrays
    train_params_jax = jnp.array(train_params_norm)
    train_targets_jax = jnp.array(train_targets)
    val_params_jax = jnp.array(val_params_norm)
    val_targets_jax = jnp.array(val_targets)

    n_outputs = train_targets.shape[1]
    print(f"  Training samples: {len(train_params_jax)}")
    print(f"  Validation samples: {len(val_params_jax)}")
    print(f"  Input dimension: {train_params_jax.shape[1]}")
    print(f"  Output dimension: {n_outputs} ({pca_amplitude.n_components} amp + {pca_phase.n_components} phase)")

    # ==========================================================================
    # Step 3: Initialize network
    # ==========================================================================
    print("\n[3/4] Initializing network...")

    model = EmulatorMLP(
        n_hidden=n_hidden,
        n_units=n_units,
        n_outputs=n_outputs,
    )

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, learning_rate, input_dim=3)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"  Architecture: {n_hidden} hidden layers x {n_units} units")
    print(f"  Total parameters: {n_params:,}")

    # ==========================================================================
    # Step 4: Training loop
    # ==========================================================================
    print("\n[4/4] Training...")

    initial_train_loss = float(eval_loss(state, train_params_jax, train_targets_jax))
    initial_val_loss = float(eval_loss(state, val_params_jax, val_targets_jax))
    print(f"  Initial train loss: {initial_train_loss:.6f}")
    print(f"  Initial val loss: {initial_val_loss:.6f}")

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []
    start_time = time.time()

    for epoch in tqdm(range(n_epochs), desc="Training"):
        rng, epoch_rng = jax.random.split(rng)
        state, train_loss = train_epoch(
            state, train_params_jax, train_targets_jax, batch_size, epoch_rng
        )
        val_loss = float(eval_loss(state, val_params_jax, val_targets_jax))

        train_losses.append(float(train_loss))
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = state
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 20 == 0:
            tqdm.write(f"  Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time
    state = best_state  # Use best model

    final_train_loss = float(eval_loss(state, train_params_jax, train_targets_jax))
    final_val_loss = float(eval_loss(state, val_params_jax, val_targets_jax))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(train_losses):.2f}s/epoch)")
    print(f"  Epochs: {len(train_losses)}")
    print(f"  Final train loss: {final_train_loss:.6f}")
    print(f"  Final val loss: {final_val_loss:.6f}")
    print(f"  Improvement: {initial_val_loss / final_val_loss:.1f}x")

    # ==========================================================================
    # Create emulator and save
    # ==========================================================================
    emulator = GWEmulator(
        network=model,
        network_params=state.params,
        pca_amplitude=pca_amplitude,
        pca_phase=pca_phase,
        frequency_grid=data["frequency_grid"],
        params_mean=params_mean,
        params_std=params_std,
    )

    model_path = output_dir / "emulator_22mode.pkl"
    emulator.save(model_path)
    print(f"\nSaved emulator to: {model_path}")

    # ==========================================================================
    # Generate training plots
    # ==========================================================================
    print("\nGenerating training plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Training curves
    ax = axes[0, 0]
    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, label='Train', linewidth=2)
    ax.semilogy(epochs, val_losses, label='Validation', linewidth=2)
    ax.axhline(best_val_loss, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_val_loss:.2e}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (PCA coefficients)')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: PCA explained variance
    ax = axes[0, 1]
    ax.bar(range(pca_amplitude.n_components),
           pca_amplitude.explained_variance_ratio_,
           alpha=0.7, label='Amplitude')
    ax.bar(range(pca_phase.n_components),
           pca_phase.explained_variance_ratio_,
           alpha=0.7, label='Phase')
    ax.set_xlabel('PCA Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Component Importance')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 3: Prediction vs target (PCA coefficients)
    ax = axes[1, 0]
    predictions = np.array(state.apply_fn(state.params, val_params_jax))
    targets = np.array(val_targets_jax)
    # Sample subset for clarity
    n_plot = min(500, len(predictions))
    for i in range(min(5, n_outputs)):
        ax.scatter(targets[:n_plot, i], predictions[:n_plot, i],
                  alpha=0.3, s=5, label=f'Coeff {i}')
    lims = [min(targets[:n_plot, :5].min(), predictions[:n_plot, :5].min()),
            max(targets[:n_plot, :5].max(), predictions[:n_plot, :5].max())]
    ax.plot(lims, lims, 'k--', linewidth=1)
    ax.set_xlabel('Target PCA Coefficient')
    ax.set_ylabel('Predicted PCA Coefficient')
    ax.set_title('PCA Coefficient Predictions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Per-component MSE
    ax = axes[1, 1]
    per_component_mse = np.mean((predictions - targets) ** 2, axis=0)
    ax.bar(range(n_outputs), per_component_mse, alpha=0.7)
    ax.axvline(pca_amplitude.n_components - 0.5, color='r', linestyle='--',
               label='Amp|Phase boundary')
    ax.set_xlabel('PCA Component Index')
    ax.set_ylabel('MSE')
    ax.set_title('Per-Component MSE')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_summary.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/training_summary.png")

    # History for return
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "n_epochs": len(train_losses),
        "elapsed_time": elapsed,
    }

    return emulator, history


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train waveform emulator")
    parser.add_argument("--data", type=str, default="data/waveforms_22mode.h5",
                        help="Input HDF5 file")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--n_hidden", type=int, default=4,
                        help="Number of hidden layers")
    parser.add_argument("--n_units", type=int, default=512,
                        help="Units per hidden layer")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    data = load_training_data(args.data)

    # Train
    emulator, history = train_emulator(
        data=data,
        n_hidden=args.n_hidden,
        n_units=args.n_units,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        output_dir=args.output,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
