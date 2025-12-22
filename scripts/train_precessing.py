#!/usr/bin/env python
"""
Train the gravitational waveform emulator for PRECESSING spins.

7D parameter space: eta, chi1, chi2, tilt1, tilt2, phi12, phi_jl

This requires:
- More training data (50k+ samples recommended)
- Potentially deeper/wider networks
- More PCA components

Usage:
    python scripts/train_precessing.py --data data/waveforms_precessing.h5
    python scripts/train_precessing.py --n_units 1024 --n_hidden 6  # Larger network
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

jax.config.update("jax_enable_x64", True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.emulator import WaveformPCA, EmulatorMLP
from jim_emulators.emulator.emulator import create_train_state, train_step, eval_loss


# =============================================================================
# Data Loading
# =============================================================================

def load_training_data(path: str):
    """Load precessing training data from HDF5 file."""
    with h5py.File(path, "r") as f:
        freqs = f["freqs"][:]
        modes = list(f.attrs["modes"])
        param_names = list(f.attrs["param_names"])
        n_params = int(f.attrs["n_params"])

        data = {
            "train_params": f["train/parameters"][:],
            "val_params": f["validation/parameters"][:],
            "freqs": freqs,
            "modes": modes,
            "param_names": param_names,
            "n_params": n_params,
            "f_min": float(f.attrs.get("f_min", freqs[0])),
            "f_max": float(f.attrs.get("f_max", freqs[-1])),
            "delta_f": float(f.attrs.get("delta_f", freqs[1] - freqs[0])),
            "M_ref": float(f.attrs.get("M_ref", 50.0)),
        }

        for mode in modes:
            data[f"train_amp_{mode}"] = f[f"train/log_amplitude_{mode}"][:]
            data[f"val_amp_{mode}"] = f[f"validation/log_amplitude_{mode}"][:]

    return data


# =============================================================================
# Training
# =============================================================================

def train_epoch(state, train_params, train_targets, batch_size, rng):
    """Train for one epoch."""
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
    input_dim: int,
):
    """Train a single network."""
    n_outputs = train_targets.shape[1]

    # Normalize targets
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
    state = create_train_state(init_rng, model, learning_rate, input_dim=input_dim)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"  [{name}] Architecture: {input_dim}D → {n_hidden}x{n_units} → {n_outputs}")
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
    n_hidden: int = 6,          # Deeper for 7D
    n_units: int = 1024,        # Wider for 7D
    batch_size: int = 512,
    n_epochs: int = 300,
    learning_rate: float = 1e-3,
    patience: int = 50,
    pca_variance: float = 0.9999,
    output_dir: str = "outputs",
):
    """Train the precessing waveform emulator."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_modes = data["modes"]
    if modes_to_train is None:
        modes_to_train = all_modes

    n_params = data["n_params"]
    param_names = data["param_names"]

    print("=" * 60)
    print("Training PRECESSING Waveform Emulator")
    print("=" * 60)
    print(f"Parameter space: {n_params}D ({', '.join(param_names)})")
    print(f"Modes to train: {modes_to_train}")
    print(f"Network: {n_hidden} layers x {n_units} units")

    # Normalize input parameters
    print("\n[1/3] Preparing input parameters...")

    params_mean = np.mean(data["train_params"], axis=0)
    params_std = np.std(data["train_params"], axis=0)

    train_params_norm = (data["train_params"] - params_mean) / params_std
    val_params_norm = (data["val_params"] - params_mean) / params_std

    train_params_jax = jnp.array(train_params_norm)
    val_params_jax = jnp.array(val_params_norm)

    print(f"  Training samples: {len(train_params_jax)}")
    print(f"  Validation samples: {len(val_params_jax)}")
    print(f"  Input dimensions: {n_params}")

    # Train networks
    print("\n[2/3] Training networks...")

    rng = jax.random.PRNGKey(42)
    start_time = time.time()

    mode_results = {}
    pca_models = {}

    for mode in modes_to_train:
        print(f"\n{'='*40}")
        print(f"Mode {mode}")
        print(f"{'='*40}")

        train_amp = data[f"train_amp_{mode}"]
        val_amp = data[f"val_amp_{mode}"]

        # PCA - may need more components for precessing
        pca = WaveformPCA(explained_variance_target=pca_variance)
        pca.fit(train_amp)
        pca_models[mode] = pca

        print(f"  PCA: {pca.n_components} components ({np.sum(pca.explained_variance_ratio_):.4f} variance)")

        train_coeffs = jnp.array(pca.transform(train_amp))
        val_coeffs = jnp.array(pca.transform(val_amp))

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
            input_dim=n_params,
        )
        mode_results[mode] = result

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s")
    for mode in modes_to_train:
        print(f"  Mode {mode} best val loss: {mode_results[mode]['best_val_loss']:.6f}")

    # Save
    print("\n[3/3] Saving emulator...")

    import pickle

    emulator_data = {
        "n_hidden": n_hidden,
        "n_units": n_units,
        "params_mean": params_mean,
        "params_std": params_std,
        "param_names": param_names,
        "n_params": n_params,
        "freqs": data["freqs"],
        "f_min": data["f_min"],
        "f_max": data["f_max"],
        "delta_f": data["delta_f"],
        "M_ref": data["M_ref"],
        "modes": modes_to_train,
        "precessing": True,
    }

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

    model_path = output_dir / "emulator_precessing.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(emulator_data, f)
    print(f"  Saved to: {model_path}")

    # Plots
    print("\nGenerating training plots...")

    n_modes = len(modes_to_train)
    fig, axes = plt.subplots(n_modes, 2, figsize=(12, 4 * n_modes))
    if n_modes == 1:
        axes = axes.reshape(1, -1)

    for i, mode in enumerate(modes_to_train):
        result = mode_results[mode]
        pca = pca_models[mode]

        ax = axes[i, 0]
        epochs = range(1, len(result["train_losses"]) + 1)
        ax.semilogy(epochs, result["train_losses"], label='Train', linewidth=2)
        ax.semilogy(epochs, result["val_losses"], label='Validation', linewidth=2)
        ax.axhline(result["best_val_loss"], color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'Mode {mode} (best: {result["best_val_loss"]:.2e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.bar(range(min(20, pca.n_components)), pca.explained_variance_ratio_[:20], alpha=0.7)
        ax.set_xlabel('PCA Component')
        ax.set_ylabel('Explained Variance')
        ax.set_title(f'Mode {mode} PCA ({pca.n_components} components)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_precessing.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/training_precessing.png")

    return emulator_data, mode_results


def main():
    parser = argparse.ArgumentParser(description="Train precessing waveform emulator")
    parser.add_argument("--data", type=str, default="data/waveforms_precessing.h5")
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--mode", type=str, default=None, help="Train single mode")
    parser.add_argument("--n_hidden", type=int, default=6)
    parser.add_argument("--n_units", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--pca_variance", type=float, default=0.9999)
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    data = load_training_data(args.data)
    print(f"  Parameters: {data['param_names']}")
    print(f"  Modes: {data['modes']}")

    modes_to_train = [args.mode] if args.mode else None

    train_emulator(
        data=data,
        modes_to_train=modes_to_train,
        n_hidden=args.n_hidden,
        n_units=args.n_units,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        pca_variance=args.pca_variance,
        output_dir=args.output,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
