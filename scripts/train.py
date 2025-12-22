#!/usr/bin/env python
"""
Train the gravitational waveform emulator.

This script trains TWO separate neural networks:
1. Amplitude network: params → amplitude PCA coefficients
2. Phase network: params → phase PCA coefficients

Pipeline (following implementation/training-pipeline.tex):
1. Per-frequency normalization (with σ floor) on outputs
2. PCA on normalized data (no double standardization)
3. Coefficient standardization for training stability
4. Train separate networks for amplitude and phase
5. Save all components for inference

Usage:
    python scripts/train.py                           # Default config
    python scripts/train.py --epochs 200 --batch_size 512
    python scripts/train.py --data data/waveforms_22mode.h5
"""

import argparse
import time
from pathlib import Path
import pickle

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

from jim_emulators.emulator import (
    WaveformPCA,
    EmulatorMLP,
    PerFrequencyNormalizer,
    CoefficientStandardizer,
    InputNormalizer,
    create_train_state,
    train_step,
    eval_loss,
)


# =============================================================================
# Data Loading
# =============================================================================

def load_training_data(path: str):
    """Load training data from HDF5 file."""
    with h5py.File(path, "r") as f:
        data = {
            "train_params": f["train/parameters"][:],
            "train_log_amp": f["train/log_amplitude"][:],
            "train_phase": f["train/phase"][:],
            "val_params": f["validation/parameters"][:],
            "val_log_amp": f["validation/log_amplitude"][:],
            "val_phase": f["validation/phase"][:],
            "frequency_grid": f["frequency_grid"][:],
            # Metadata
            "eta_bounds": (f.attrs["param_bounds_eta"][0], f.attrs["param_bounds_eta"][1]),
            "chi_bounds": (f.attrs["param_bounds_chi1z"][0], f.attrs["param_bounds_chi1z"][1]),
        }
    return data


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(state, train_params, train_targets, batch_size, rng):
    """Train for one epoch with shuffled batches."""
    n_samples = len(train_params)
    # Use smaller batch size if we don't have enough samples
    effective_batch_size = min(batch_size, n_samples)
    n_batches = max(1, n_samples // effective_batch_size)

    perm = jax.random.permutation(rng, n_samples)
    train_params = train_params[perm]
    train_targets = train_targets[perm]

    epoch_loss = 0.0
    for i in range(n_batches):
        batch_params = train_params[i * effective_batch_size:(i + 1) * effective_batch_size]
        batch_targets = train_targets[i * effective_batch_size:(i + 1) * effective_batch_size]
        state, loss = train_step(state, batch_params, batch_targets)
        epoch_loss += loss

    return state, epoch_loss / n_batches


def train_single_network(
    name: str,
    train_params: jnp.ndarray,
    train_targets: jnp.ndarray,
    val_params: jnp.ndarray,
    val_targets: jnp.ndarray,
    n_hidden: int,
    n_units: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    patience: int,
    rng: jax.random.PRNGKey,
):
    """
    Train a single network (amplitude or phase).

    Returns trained state and training history.
    """
    n_outputs = train_targets.shape[1]

    # Create model
    model = EmulatorMLP(
        n_hidden=n_hidden,
        n_units=n_units,
        n_outputs=n_outputs,
    )

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, learning_rate, input_dim=3)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"  [{name}] Architecture: {n_hidden}x{n_units} → {n_outputs} outputs")
    print(f"  [{name}] Parameters: {n_params:,}")

    initial_loss = float(eval_loss(state, val_params, val_targets))
    print(f"  [{name}] Initial val loss: {initial_loss:.6f}")

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(n_epochs), desc=f"Training {name}"):
        rng, epoch_rng = jax.random.split(rng)
        state, train_loss = train_epoch(
            state, train_params, train_targets, batch_size, epoch_rng
        )
        val_loss = float(eval_loss(state, val_params, val_targets))

        train_losses.append(float(train_loss))
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = state
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            tqdm.write(f"  [{name}] Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  [{name}] Early stopping at epoch {epoch+1}")
            break

    final_loss = float(eval_loss(best_state, val_params, val_targets))
    print(f"  [{name}] Final val loss: {final_loss:.6f} (improvement: {initial_loss/final_loss:.1f}x)")

    return {
        "model": model,
        "state": best_state,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "n_epochs_trained": len(train_losses),
    }


def train_emulator(
    data: dict,
    n_hidden: int = 4,
    n_units: int = 512,
    batch_size: int = 256,
    n_epochs: int = 500,
    learning_rate: float = 1e-3,
    patience: int = 50,
    output_dir: str = "outputs",
    sigma_floor: float = 1e-6,
):
    """
    Train the waveform emulator with the proper normalization pipeline.

    Pipeline:
    1. Per-frequency normalization (with σ floor)
    2. PCA on normalized data (no double standardization)
    3. Coefficient standardization
    4. Train separate networks for amplitude and phase
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Waveform Emulator")
    print("=" * 60)
    print(f"  Training samples: {len(data['train_params'])}")
    print(f"  Validation samples: {len(data['val_params'])}")
    print(f"  Frequency bins: {len(data['frequency_grid'])}")

    # ==========================================================================
    # Step 1: Per-frequency normalization (with σ floor)
    # ==========================================================================
    print("\n[1/5] Per-frequency normalization...")

    amp_normalizer = PerFrequencyNormalizer(sigma_floor=sigma_floor)
    phase_normalizer = PerFrequencyNormalizer(sigma_floor=sigma_floor)

    # Fit on training data
    train_amp_norm = amp_normalizer.fit_transform(data["train_log_amp"])
    train_phase_norm = phase_normalizer.fit_transform(data["train_phase"])

    # Transform validation data
    val_amp_norm = amp_normalizer.transform(data["val_log_amp"])
    val_phase_norm = phase_normalizer.transform(data["val_phase"])

    print(f"  Amplitude σ range: [{amp_normalizer.std_.min():.2e}, {amp_normalizer.std_.max():.2e}]")
    print(f"  Phase σ range: [{phase_normalizer.std_.min():.2e}, {phase_normalizer.std_.max():.2e}]")

    # Verify normalization
    print(f"  Normalized amplitude: mean={train_amp_norm.mean():.6f}, std={train_amp_norm.std():.3f}")
    print(f"  Normalized phase: mean={train_phase_norm.mean():.6f}, std={train_phase_norm.std():.3f}")

    # ==========================================================================
    # Step 2: PCA on normalized data (no double standardization)
    # ==========================================================================
    print("\n[2/5] PCA compression on normalized data...")

    pca_amplitude = WaveformPCA(explained_variance_target=0.9999)
    pca_phase = WaveformPCA(explained_variance_target=0.9999)

    # Fit PCA on normalized data
    train_amp_coeffs = pca_amplitude.fit_transform(train_amp_norm)
    train_phase_coeffs = pca_phase.fit_transform(train_phase_norm)

    # Transform validation
    val_amp_coeffs = pca_amplitude.transform(val_amp_norm)
    val_phase_coeffs = pca_phase.transform(val_phase_norm)

    print(f"  Amplitude PCA: {pca_amplitude.n_components} components "
          f"({np.sum(pca_amplitude.explained_variance_ratio_):.6f} variance)")
    print(f"  Phase PCA: {pca_phase.n_components} components "
          f"({np.sum(pca_phase.explained_variance_ratio_):.6f} variance)")

    # Check reconstruction error (on normalized data)
    amp_mse, _ = pca_amplitude.reconstruction_error(train_amp_norm)
    phase_mse, _ = pca_phase.reconstruction_error(train_phase_norm)
    print(f"  Amplitude reconstruction MSE: {amp_mse:.2e}")
    print(f"  Phase reconstruction MSE: {phase_mse:.2e}")

    # ==========================================================================
    # Step 3: PCA coefficient standardization
    # ==========================================================================
    print("\n[3/5] Coefficient standardization...")

    amp_coeff_std = CoefficientStandardizer()
    phase_coeff_std = CoefficientStandardizer()

    # Fit and transform training coefficients
    train_amp_coeffs_std = amp_coeff_std.fit_transform(train_amp_coeffs)
    train_phase_coeffs_std = phase_coeff_std.fit_transform(train_phase_coeffs)

    # Transform validation coefficients
    val_amp_coeffs_std = amp_coeff_std.transform(val_amp_coeffs)
    val_phase_coeffs_std = phase_coeff_std.transform(val_phase_coeffs)

    print(f"  Amp coefficients: mean={train_amp_coeffs_std.mean():.6f}, std={train_amp_coeffs_std.std():.3f}")
    print(f"  Phase coefficients: mean={train_phase_coeffs_std.mean():.6f}, std={train_phase_coeffs_std.std():.3f}")

    # ==========================================================================
    # Step 4: Input normalization
    # ==========================================================================
    print("\n[4/5] Input parameter normalization...")

    input_normalizer = InputNormalizer(
        eta_bounds=data["eta_bounds"],
        chi_bounds=data["chi_bounds"],
    )

    train_params_norm = input_normalizer.transform(data["train_params"])
    val_params_norm = input_normalizer.transform(data["val_params"])

    print(f"  Input range: [{train_params_norm.min():.3f}, {train_params_norm.max():.3f}]")

    # Convert to JAX arrays
    train_params_jax = jnp.array(train_params_norm)
    val_params_jax = jnp.array(val_params_norm)
    train_amp_jax = jnp.array(train_amp_coeffs_std)
    train_phase_jax = jnp.array(train_phase_coeffs_std)
    val_amp_jax = jnp.array(val_amp_coeffs_std)
    val_phase_jax = jnp.array(val_phase_coeffs_std)

    # ==========================================================================
    # Step 5: Train networks
    # ==========================================================================
    print("\n[5/5] Training networks...")

    rng = jax.random.PRNGKey(42)
    start_time = time.time()

    # Train amplitude network
    print("\n--- Amplitude Network ---")
    rng, amp_rng = jax.random.split(rng)
    amp_result = train_single_network(
        name="Amplitude",
        train_params=train_params_jax,
        train_targets=train_amp_jax,
        val_params=val_params_jax,
        val_targets=val_amp_jax,
        n_hidden=n_hidden,
        n_units=n_units,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        patience=patience,
        rng=amp_rng,
    )

    # Train phase network
    print("\n--- Phase Network ---")
    rng, phase_rng = jax.random.split(rng)
    phase_result = train_single_network(
        name="Phase",
        train_params=train_params_jax,
        train_targets=train_phase_jax,
        val_params=val_params_jax,
        val_targets=val_phase_jax,
        n_hidden=n_hidden,
        n_units=n_units,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        patience=patience,
        rng=phase_rng,
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Amplitude best val loss: {amp_result['best_val_loss']:.6f}")
    print(f"  Phase best val loss: {phase_result['best_val_loss']:.6f}")

    # ==========================================================================
    # Save emulator
    # ==========================================================================
    print("\nSaving emulator...")

    emulator_data = {
        # Network configs
        "n_hidden": n_hidden,
        "n_units": n_units,

        # Amplitude network
        "amp_n_outputs": pca_amplitude.n_components,
        "amp_params": amp_result["state"].params,

        # Phase network
        "phase_n_outputs": pca_phase.n_components,
        "phase_params": phase_result["state"].params,

        # Normalizers (per-frequency)
        "amp_normalizer": amp_normalizer.to_dict(),
        "phase_normalizer": phase_normalizer.to_dict(),

        # PCA (no internal standardization)
        "pca_amplitude": pca_amplitude.to_dict(),
        "pca_phase": pca_phase.to_dict(),

        # Coefficient standardizers
        "amp_coeff_std": amp_coeff_std.to_dict(),
        "phase_coeff_std": phase_coeff_std.to_dict(),

        # Input normalizer
        "input_normalizer": input_normalizer.to_dict(),

        # Grid
        "frequency_grid": data["frequency_grid"],

        # Training info
        "training_info": {
            "elapsed_time": elapsed,
            "amp_epochs": amp_result["n_epochs_trained"],
            "phase_epochs": phase_result["n_epochs_trained"],
            "amp_best_val_loss": amp_result["best_val_loss"],
            "phase_best_val_loss": phase_result["best_val_loss"],
        },
    }

    model_path = output_dir / "emulator_22mode.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(emulator_data, f)
    print(f"  Saved to: {model_path}")

    # ==========================================================================
    # Generate training plots
    # ==========================================================================
    print("\nGenerating training plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Amplitude training curves
    ax = axes[0, 0]
    epochs = range(1, len(amp_result["train_losses"]) + 1)
    ax.semilogy(epochs, amp_result["train_losses"], label='Train', linewidth=2)
    ax.semilogy(epochs, amp_result["val_losses"], label='Validation', linewidth=2)
    ax.axhline(amp_result["best_val_loss"], color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Amplitude Network (best: {amp_result["best_val_loss"]:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Phase training curves
    ax = axes[0, 1]
    epochs = range(1, len(phase_result["train_losses"]) + 1)
    ax.semilogy(epochs, phase_result["train_losses"], label='Train', linewidth=2)
    ax.semilogy(epochs, phase_result["val_losses"], label='Validation', linewidth=2)
    ax.axhline(phase_result["best_val_loss"], color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Phase Network (best: {phase_result["best_val_loss"]:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: PCA explained variance (cumulative)
    ax = axes[1, 0]
    amp_cumvar = np.cumsum(pca_amplitude.explained_variance_ratio_)
    phase_cumvar = np.cumsum(pca_phase.explained_variance_ratio_)
    ax.plot(range(1, len(amp_cumvar) + 1), amp_cumvar, 'o-', label='Amplitude', markersize=4)
    ax.plot(range(1, len(phase_cumvar) + 1), phase_cumvar, 's-', label='Phase', markersize=4)
    ax.axhline(0.9999, color='r', linestyle='--', alpha=0.5, label='99.99% target')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA Component Importance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Waveform reconstruction check
    ax = axes[1, 1]

    # Predict on validation set and reconstruct waveforms
    amp_pred_std = amp_result["state"].apply_fn(amp_result["state"].params, val_params_jax)
    phase_pred_std = phase_result["state"].apply_fn(phase_result["state"].params, val_params_jax)

    # Denormalize coefficients
    amp_pred_coeffs = amp_coeff_std.inverse_transform(np.array(amp_pred_std))
    phase_pred_coeffs = phase_coeff_std.inverse_transform(np.array(phase_pred_std))

    # PCA reconstruction (to normalized space)
    amp_pred_norm = pca_amplitude.inverse_transform(amp_pred_coeffs)
    phase_pred_norm = pca_phase.inverse_transform(phase_pred_coeffs)

    # Denormalize to original space
    amp_pred = amp_normalizer.inverse_transform(amp_pred_norm)
    phase_pred = phase_normalizer.inverse_transform(phase_pred_norm)

    # Compare with original
    amp_residual = np.mean((amp_pred - data["val_log_amp"]) ** 2)
    phase_residual = np.mean((phase_pred - data["val_phase"]) ** 2)

    ax.bar(['Log Amplitude', 'Phase'], [amp_residual, phase_residual], alpha=0.7)
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title(f'Full Pipeline MSE\nAmp: {amp_residual:.2e}, Phase: {phase_residual:.2e}')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_summary.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/training_summary.png")

    # ==========================================================================
    # Sample waveform comparison
    # ==========================================================================
    print("Generating sample waveform plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Pick 3 random validation samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(data["val_params"]), 3, replace=False)

    for i, idx in enumerate(sample_indices):
        # True waveform
        true_amp = data["val_log_amp"][idx]
        true_phase = data["val_phase"][idx]

        # Predicted waveform
        pred_amp = amp_pred[idx]
        pred_phase = phase_pred[idx]

        # Plot amplitude
        ax = axes[0, i]
        ax.plot(data["frequency_grid"], true_amp, 'b-', label='True', linewidth=1.5)
        ax.plot(data["frequency_grid"], pred_amp, 'r--', label='Emulated', linewidth=1.5)
        ax.set_xlabel('Mf')
        ax.set_ylabel('log₁₀|h|')
        ax.set_xscale('log')
        params = data["val_params"][idx]
        ax.set_title(f'η={params[0]:.3f}, χ₁={params[1]:.2f}, χ₂={params[2]:.2f}')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot phase
        ax = axes[1, i]
        ax.plot(data["frequency_grid"], true_phase, 'b-', label='True', linewidth=1.5)
        ax.plot(data["frequency_grid"], pred_phase, 'r--', label='Emulated', linewidth=1.5)
        ax.set_xlabel('Mf')
        ax.set_ylabel('Phase [rad]')
        ax.set_xscale('log')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "sample_waveforms.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/sample_waveforms.png")

    return emulator_data, {
        "amp": amp_result,
        "phase": phase_result,
        "elapsed_time": elapsed,
    }


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
    parser.add_argument("--epochs", type=int, default=500,
                        help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience")
    parser.add_argument("--sigma_floor", type=float, default=1e-6,
                        help="Floor for per-frequency std")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    data = load_training_data(args.data)

    # Train
    train_emulator(
        data=data,
        n_hidden=args.n_hidden,
        n_units=args.n_units,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        output_dir=args.output,
        sigma_floor=args.sigma_floor,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
