#!/usr/bin/env python
"""
Train the gravitational waveform emulator.

Trains ONE network per mode for amplitude only (phase handled separately).

Usage:
    python scripts/train.py                           # Default config
    python scripts/train.py --epochs 200 --batch_size 512
    python scripts/train.py --data data/waveforms_multimode.h5
    python scripts/train.py --mode 22                 # Train single mode
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
        freqs = f["freqs"][:]
        modes = list(f.attrs["modes"])

        data = {
            "train_params": f["train/parameters"][:],
            "val_params": f["validation/parameters"][:],
            "freqs": freqs,
            "modes": modes,
            "f_min": float(f.attrs.get("f_min", freqs[0])),
            "f_max": float(f.attrs.get("f_max", freqs[-1])),
            "delta_f": float(f.attrs.get("delta_f", freqs[1] - freqs[0])),
            "M_ref": float(f.attrs.get("M_ref", 50.0)),
        }

        # Load amplitude for each mode
        for mode in modes:
            data[f"train_amp_{mode}"] = f[f"train/log_amplitude_{mode}"][:]
            data[f"val_amp_{mode}"] = f[f"validation/log_amplitude_{mode}"][:]

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
    """Train a single network for one mode's amplitude."""
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
    modes_to_train: list = None,
    n_hidden: int = 4,
    n_units: int = 512,
    batch_size: int = 256,
    n_epochs: int = 200,
    learning_rate: float = 1e-3,
    patience: int = 30,
    output_dir: str = "outputs",
):
    """Train the waveform emulator with one network per mode."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_modes = data["modes"]
    if modes_to_train is None:
        modes_to_train = all_modes

    print("=" * 60)
    print("Training Waveform Emulator (Amplitude Only, Per-Mode)")
    print("=" * 60)
    print(f"Modes to train: {modes_to_train}")

    # ==========================================================================
    # Step 1: Normalize input parameters (shared across modes)
    # ==========================================================================
    print("\n[1/3] Preparing input parameters...")

    params_mean = np.mean(data["train_params"], axis=0)
    params_std = np.std(data["train_params"], axis=0)

    train_params_norm = (data["train_params"] - params_mean) / params_std
    val_params_norm = (data["val_params"] - params_mean) / params_std

    train_params_jax = jnp.array(train_params_norm)
    val_params_jax = jnp.array(val_params_norm)

    print(f"  Training samples: {len(train_params_jax)}")
    print(f"  Validation samples: {len(val_params_jax)}")

    # ==========================================================================
    # Step 2: Train one network per mode
    # ==========================================================================
    print("\n[2/3] Training networks...")

    rng = jax.random.PRNGKey(42)
    start_time = time.time()

    mode_results = {}
    pca_models = {}

    for mode in modes_to_train:
        print(f"\n{'='*40}")
        print(f"Mode {mode}")
        print(f"{'='*40}")

        # Fit PCA for this mode
        train_amp = data[f"train_amp_{mode}"]
        val_amp = data[f"val_amp_{mode}"]

        pca = WaveformPCA(explained_variance_target=0.9999)
        pca.fit(train_amp)
        pca_models[mode] = pca

        print(f"  PCA: {pca.n_components} components ({np.sum(pca.explained_variance_ratio_):.4f} variance)")

        # Transform to PCA coefficients
        train_coeffs = jnp.array(pca.transform(train_amp))
        val_coeffs = jnp.array(pca.transform(val_amp))

        # Train network
        rng, mode_rng = jax.random.split(rng)
        result = train_single_network(
            name=f"Mode {mode}",
            train_params=train_params_jax,
            train_targets=train_coeffs,
            val_params=val_params_jax,
            val_targets=val_coeffs,
            n_hidden=n_hidden,
            n_units=n_units,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            patience=patience,
            rng=mode_rng,
        )
        mode_results[mode] = result

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s")
    for mode in modes_to_train:
        print(f"  Mode {mode} best val loss: {mode_results[mode]['best_val_loss']:.6f}")

    # ==========================================================================
    # Step 3: Save emulator
    # ==========================================================================
    print("\n[3/3] Saving emulator...")

    import pickle

    emulator_data = {
        # Network config
        "n_hidden": n_hidden,
        "n_units": n_units,
        # Input normalization (shared)
        "params_mean": params_mean,
        "params_std": params_std,
        # Frequency grid and metadata
        "freqs": data["freqs"],
        "f_min": data["f_min"],
        "f_max": data["f_max"],
        "delta_f": data["delta_f"],
        "M_ref": data["M_ref"],
        "modes": modes_to_train,
    }

    # Per-mode data
    for mode in modes_to_train:
        result = mode_results[mode]
        pca = pca_models[mode]
        emulator_data[f"mode_{mode}"] = {
            "n_outputs": pca.n_components,
            "params": result["state"].params,
            "target_mean": np.array(result["target_mean"]),
            "target_std": np.array(result["target_std"]),
            "pca": pca.to_dict(),
        }

    model_path = output_dir / "emulator_multimode.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(emulator_data, f)
    print(f"  Saved to: {model_path}")

    # ==========================================================================
    # Generate training plots
    # ==========================================================================
    print("\nGenerating training plots...")

    n_modes = len(modes_to_train)
    fig, axes = plt.subplots(n_modes, 2, figsize=(12, 4 * n_modes))
    if n_modes == 1:
        axes = axes.reshape(1, -1)

    for i, mode in enumerate(modes_to_train):
        result = mode_results[mode]
        pca = pca_models[mode]

        # Training curves
        ax = axes[i, 0]
        epochs = range(1, len(result["train_losses"]) + 1)
        ax.semilogy(epochs, result["train_losses"], label='Train', linewidth=2)
        ax.semilogy(epochs, result["val_losses"], label='Validation', linewidth=2)
        ax.axhline(result["best_val_loss"], color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss (normalized)')
        ax.set_title(f'Mode {mode} (best: {result["best_val_loss"]:.2e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # PCA variance
        ax = axes[i, 1]
        ax.bar(range(pca.n_components), pca.explained_variance_ratio_, alpha=0.7)
        ax.set_xlabel('PCA Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title(f'Mode {mode} PCA ({pca.n_components} components)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_summary.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/training_summary.png")

    return emulator_data, mode_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train waveform emulator")
    parser.add_argument("--data", type=str, default="data/waveforms_multimode.h5",
                        help="Input HDF5 file")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--mode", type=str, default=None,
                        help="Train single mode (e.g., '22')")
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
    print(f"  Modes available: {data['modes']}")

    # Determine which modes to train
    modes_to_train = [args.mode] if args.mode else None

    # Train
    train_emulator(
        data=data,
        modes_to_train=modes_to_train,
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
