#!/usr/bin/env python
"""
Train the gravitational waveform emulator.

This script trains TWO separate neural networks:
1. Amplitude network: params → amplitude PCA coefficients
2. Phase network: params → phase PCA coefficients

This separation is important because amplitude and phase have very different
characteristics and scales.

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

from jim_emulators.emulator import WaveformPCA, EmulatorMLP
from jim_emulators.emulator.emulator import create_train_state, train_step, eval_loss


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
            # Frequency grid metadata for validation
            "f_min": float(f.attrs.get("f_min", f["frequency_grid"][0])),
            "f_max": float(f.attrs.get("f_max", f["frequency_grid"][-1])),
            "delta_f": float(f.attrs.get("delta_f", f["frequency_grid"][1] - f["frequency_grid"][0])),
            "M_ref": float(f.attrs.get("M_ref", 50.0)),
        }
    return data


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(state, train_params, train_targets, batch_size, rng):
    """Train for one epoch with shuffled batches."""
    n_samples = len(train_params)
    n_batches = n_samples // batch_size

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

    # Normalize targets for stable training
    target_mean = jnp.mean(train_targets, axis=0)
    target_std = jnp.std(train_targets, axis=0)
    target_std = jnp.where(target_std < 1e-10, 1.0, target_std)

    train_targets_norm = (train_targets - target_mean) / target_std
    val_targets_norm = (val_targets - target_mean) / target_std

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

    initial_loss = float(eval_loss(state, val_params, val_targets_norm))
    print(f"  [{name}] Initial val loss: {initial_loss:.6f}")

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(n_epochs), desc=f"Training {name}"):
        rng, epoch_rng = jax.random.split(rng)
        state, train_loss = train_epoch(
            state, train_params, train_targets_norm, batch_size, epoch_rng
        )
        val_loss = float(eval_loss(state, val_params, val_targets_norm))

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

    final_loss = float(eval_loss(best_state, val_params, val_targets_norm))
    print(f"  [{name}] Final val loss: {final_loss:.6f} (improvement: {initial_loss/final_loss:.1f}x)")

    return {
        "model": model,
        "state": best_state,
        "target_mean": target_mean,
        "target_std": target_std,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }


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
    Train the waveform emulator with separate amplitude and phase networks.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Waveform Emulator (Separate Amp/Phase Networks)")
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

    val_amp_coeffs = pca_amplitude.transform(data["val_log_amp"])
    val_phase_coeffs = pca_phase.transform(data["val_phase"])

    # Normalize input parameters
    params_mean = np.mean(data["train_params"], axis=0)
    params_std = np.std(data["train_params"], axis=0)

    train_params_norm = (data["train_params"] - params_mean) / params_std
    val_params_norm = (data["val_params"] - params_mean) / params_std

    # Convert to JAX arrays
    train_params_jax = jnp.array(train_params_norm)
    val_params_jax = jnp.array(val_params_norm)
    train_amp_jax = jnp.array(train_amp_coeffs)
    train_phase_jax = jnp.array(train_phase_coeffs)
    val_amp_jax = jnp.array(val_amp_coeffs)
    val_phase_jax = jnp.array(val_phase_coeffs)

    print(f"  Training samples: {len(train_params_jax)}")
    print(f"  Validation samples: {len(val_params_jax)}")
    print(f"  Amplitude PCA coeffs: {train_amp_jax.shape[1]}")
    print(f"  Phase PCA coeffs: {train_phase_jax.shape[1]}")

    # ==========================================================================
    # Step 3: Train SEPARATE networks for amplitude and phase
    # ==========================================================================
    print("\n[3/4] Training networks...")

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
    # Step 4: Save emulator
    # ==========================================================================
    print("\n[4/4] Saving emulator...")

    import pickle

    emulator_data = {
        # Network configs
        "n_hidden": n_hidden,
        "n_units": n_units,
        # Amplitude network
        "amp_n_outputs": pca_amplitude.n_components,
        "amp_params": amp_result["state"].params,
        "amp_target_mean": np.array(amp_result["target_mean"]),
        "amp_target_std": np.array(amp_result["target_std"]),
        # Phase network
        "phase_n_outputs": pca_phase.n_components,
        "phase_params": phase_result["state"].params,
        "phase_target_mean": np.array(phase_result["target_mean"]),
        "phase_target_std": np.array(phase_result["target_std"]),
        # PCA
        "pca_amplitude": pca_amplitude.to_dict(),
        "pca_phase": pca_phase.to_dict(),
        # Normalization
        "params_mean": params_mean,
        "params_std": params_std,
        # Frequency grid and metadata for validation
        "frequency_grid": data["frequency_grid"],
        "f_min": data["f_min"],
        "f_max": data["f_max"],
        "delta_f": data["delta_f"],
        "M_ref": data["M_ref"],
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
    ax.set_ylabel('MSE Loss (normalized)')
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
    ax.set_ylabel('MSE Loss (normalized)')
    ax.set_title(f'Phase Network (best: {phase_result["best_val_loss"]:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: PCA explained variance
    ax = axes[1, 0]
    ax.bar(range(pca_amplitude.n_components),
           pca_amplitude.explained_variance_ratio_,
           alpha=0.7, label='Amplitude')
    ax.bar(np.arange(pca_phase.n_components) + 0.4,
           pca_phase.explained_variance_ratio_,
           alpha=0.7, label='Phase')
    ax.set_xlabel('PCA Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Component Importance')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 4: Prediction quality check
    ax = axes[1, 1]

    # Quick prediction check on validation set
    amp_pred_norm = amp_result["state"].apply_fn(amp_result["state"].params, val_params_jax)
    amp_pred = amp_pred_norm * amp_result["target_std"] + amp_result["target_mean"]
    amp_true = val_amp_jax

    phase_pred_norm = phase_result["state"].apply_fn(phase_result["state"].params, val_params_jax)
    phase_pred = phase_pred_norm * phase_result["target_std"] + phase_result["target_mean"]
    phase_true = val_phase_jax

    # Reconstruct waveforms and compute residuals
    amp_recon_pred = pca_amplitude.inverse_transform(np.array(amp_pred))
    amp_recon_true = pca_amplitude.inverse_transform(np.array(amp_true))
    phase_recon_pred = pca_phase.inverse_transform(np.array(phase_pred))
    phase_recon_true = pca_phase.inverse_transform(np.array(phase_true))

    amp_residual = np.mean((amp_recon_pred - amp_recon_true) ** 2)
    phase_residual = np.mean((phase_recon_pred - phase_recon_true) ** 2)

    ax.bar(['Amplitude', 'Phase'], [amp_residual, phase_residual], alpha=0.7)
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title(f'Waveform Reconstruction MSE\nAmp: {amp_residual:.2e}, Phase: {phase_residual:.2e}')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_summary.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/training_summary.png")

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
    train_emulator(
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
