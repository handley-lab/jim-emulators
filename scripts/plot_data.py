#!/usr/bin/env python
"""
Plot training data to verify waveform generation quality.

Usage:
    python scripts/plot_data.py                           # Default: data/waveforms_multimode.h5
    python scripts/plot_data.py --input data/other.h5     # Custom file
    python scripts/plot_data.py --n_samples 20            # More sample waveforms
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def plot_sample_waveforms(f, n_samples=10, output_dir="figures/data_check"):
    """Plot a selection of waveforms showing amplitude for all modes."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    freqs = f["freqs"][:]
    params = f["train/parameters"][:]
    modes = list(f.attrs["modes"])

    # Select samples spanning the parameter space
    n_total = len(params)
    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)

    # Create figure with one subplot per mode
    n_modes = len(modes)
    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 3 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    # Color by eta (mass ratio)
    eta_vals = params[indices, 0]
    colors = plt.cm.viridis((eta_vals - eta_vals.min()) / (eta_vals.max() - eta_vals.min() + 1e-10))

    for ax, mode in zip(axes, modes):
        log_amp = f[f"train/log_amplitude_{mode}"][:]
        for i, idx in enumerate(indices):
            ax.plot(freqs, log_amp[idx], color=colors[i], alpha=0.7, lw=0.8)
        ax.set_ylabel(f"log₁₀(A) [{mode}]")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"Sample Waveforms (n={n_samples})")
    axes[-1].set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_waveforms.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/sample_waveforms.png")


def plot_parameter_distribution(f, output_dir="figures/data_check"):
    """Plot parameter space coverage."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    params_train = f["train/parameters"][:]
    params_val = f["validation/parameters"][:]

    param_names = ["η (mass ratio)", "χ₁z (spin 1)", "χ₂z (spin 2)"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(params_train[:, i], bins=50, alpha=0.7, label="Train", density=True)
        ax.hist(params_val[:, i], bins=30, alpha=0.7, label="Validation", density=True)
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Parameter Space Coverage (LHS Sampling)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_distribution.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/parameter_distribution.png")


def plot_amplitude_stats(f, output_dir="figures/data_check"):
    """Plot statistics of amplitude across the dataset for each mode."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    freqs = f["freqs"][:]
    modes = list(f.attrs["modes"])

    n_modes = len(modes)
    fig, axes = plt.subplots(n_modes, 2, figsize=(12, 3 * n_modes))
    if n_modes == 1:
        axes = axes.reshape(1, -1)

    for i, mode in enumerate(modes):
        log_amp = f[f"train/log_amplitude_{mode}"][:]

        # Compute statistics
        amp_mean = np.mean(log_amp, axis=0)
        amp_std = np.std(log_amp, axis=0)
        amp_min = np.min(log_amp, axis=0)
        amp_max = np.max(log_amp, axis=0)

        # Mean ± std
        ax = axes[i, 0]
        ax.fill_between(freqs, amp_mean - amp_std, amp_mean + amp_std, alpha=0.3, label="±1σ")
        ax.plot(freqs, amp_mean, "k-", lw=1.5, label="Mean")
        ax.set_ylabel(f"log₁₀(A) [{mode}]")
        ax.set_xscale("log")
        ax.set_title(f"Mode {mode}: Mean ± Std")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Full range
        ax = axes[i, 1]
        ax.fill_between(freqs, amp_min, amp_max, alpha=0.3, label="Min-Max")
        ax.plot(freqs, amp_mean, "k-", lw=1.5, label="Mean")
        ax.set_ylabel(f"log₁₀(A) [{mode}]")
        ax.set_xscale("log")
        ax.set_title(f"Mode {mode}: Full Range")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Frequency (Hz)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/amplitude_stats.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/amplitude_stats.png")


def plot_extreme_cases(f, output_dir="figures/data_check"):
    """Plot waveforms at extreme parameter values."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    freqs = f["freqs"][:]
    params = f["train/parameters"][:]
    modes = list(f.attrs["modes"])

    # Find extremes
    extremes = {
        "Low η (high q)": np.argmin(params[:, 0]),
        "High η (q~1)": np.argmax(params[:, 0]),
        "High χ₁z": np.argmax(params[:, 1]),
        "Low χ₁z": np.argmin(params[:, 1]),
        "Both spins +": np.argmax(params[:, 1] + params[:, 2]),
        "Both spins -": np.argmin(params[:, 1] + params[:, 2]),
    }

    n_modes = len(modes)
    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 3 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        log_amp = f[f"train/log_amplitude_{mode}"][:]
        for name, idx in extremes.items():
            eta, chi1z, chi2z = params[idx]
            label = f"{name}: η={eta:.3f}, χ₁={chi1z:.2f}, χ₂={chi2z:.2f}"
            ax.plot(freqs, log_amp[idx], lw=1.2, label=label if mode == modes[0] else None)
        ax.set_ylabel(f"log₁₀(A) [{mode}]")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Extreme Cases in Parameter Space")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/extreme_cases.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/extreme_cases.png")


def plot_mode_comparison(f, output_dir="figures/data_check"):
    """Compare amplitudes across modes for a few samples."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    freqs = f["freqs"][:]
    params = f["train/parameters"][:]
    modes = list(f.attrs["modes"])

    # Pick 3 samples: low eta, mid eta, high eta
    eta_sorted = np.argsort(params[:, 0])
    indices = [eta_sorted[0], eta_sorted[len(eta_sorted) // 2], eta_sorted[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, idx in zip(axes, indices):
        eta, chi1z, chi2z = params[idx]
        for mode in modes:
            log_amp = f[f"train/log_amplitude_{mode}"][idx]
            ax.plot(freqs, log_amp, label=mode, lw=1.2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("log₁₀(Amplitude)")
        ax.set_xscale("log")
        ax.set_title(f"η={eta:.3f}, χ₁={chi1z:.2f}, χ₂={chi2z:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Mode Comparison Across Parameter Space")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mode_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/mode_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Plot training data for verification")
    parser.add_argument("--input", type=str, default="data/waveforms_multimode.h5",
                        help="Input HDF5 file")
    parser.add_argument("--output_dir", type=str, default="figures/data_check",
                        help="Output directory for plots")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of sample waveforms to plot")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    with h5py.File(args.input, "r") as f:
        print(f"  Train samples: {f['train/parameters'].shape[0]}")
        print(f"  Validation samples: {f['validation/parameters'].shape[0]}")
        print(f"  Frequency points: {f['freqs'].shape[0]}")
        print(f"  Modes: {list(f.attrs['modes'])}")

        plot_sample_waveforms(f, n_samples=args.n_samples, output_dir=args.output_dir)
        plot_parameter_distribution(f, output_dir=args.output_dir)
        plot_amplitude_stats(f, output_dir=args.output_dir)
        plot_extreme_cases(f, output_dir=args.output_dir)
        plot_mode_comparison(f, output_dir=args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
