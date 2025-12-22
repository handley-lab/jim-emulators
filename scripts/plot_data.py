#!/usr/bin/env python
"""
Plot training data to verify waveform generation quality.

Usage:
    python scripts/plot_data.py                           # Default: data/waveforms_22mode.h5
    python scripts/plot_data.py --input data/other.h5     # Custom file
    python scripts/plot_data.py --n_samples 20            # More sample waveforms
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def plot_sample_waveforms(f, n_samples=10, output_dir="figures/data_check"):
    """Plot a selection of waveforms showing amplitude and phase."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    freqs = f["frequency_grid"][:]
    params = f["train/parameters"][:]
    log_amp = f["train/log_amplitude"][:]
    phase = f["train/phase"][:]

    # Select samples spanning the parameter space
    n_total = len(params)
    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)

    # Create figure with amplitude and phase subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Color by eta (mass ratio)
    eta_vals = params[indices, 0]
    colors = plt.cm.viridis((eta_vals - eta_vals.min()) / (eta_vals.max() - eta_vals.min()))

    for i, idx in enumerate(indices):
        eta, chi1z, chi2z = params[idx]
        label = f"η={eta:.3f}, χ₁={chi1z:.2f}, χ₂={chi2z:.2f}"

        axes[0].plot(freqs, log_amp[idx], color=colors[i], alpha=0.7, lw=0.8)
        axes[1].plot(freqs, phase[idx], color=colors[i], alpha=0.7, lw=0.8, label=label)

    axes[0].set_ylabel("log₁₀(Amplitude)")
    axes[0].set_title(f"Sample Waveforms (n={n_samples})")
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=6, ncol=2)

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

    # Corner plot (2D projections)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i == j:
                ax.hist(params_train[:, i], bins=50, alpha=0.7)
                ax.set_xlabel(param_names[i])
            elif i > j:
                ax.scatter(params_train[:, j], params_train[:, i],
                          alpha=0.1, s=1, c="C0")
                ax.set_xlabel(param_names[j])
                ax.set_ylabel(param_names[i])
            else:
                ax.axis("off")

    plt.suptitle("Parameter Space Coverage (2D Projections)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_corner.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/parameter_corner.png")


def plot_amplitude_phase_stats(f, output_dir="figures/data_check"):
    """Plot statistics of amplitude and phase across the dataset."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    freqs = f["frequency_grid"][:]
    log_amp = f["train/log_amplitude"][:]
    phase = f["train/phase"][:]

    # Compute statistics
    amp_mean = np.mean(log_amp, axis=0)
    amp_std = np.std(log_amp, axis=0)
    amp_min = np.min(log_amp, axis=0)
    amp_max = np.max(log_amp, axis=0)

    phase_mean = np.mean(phase, axis=0)
    phase_std = np.std(phase, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Amplitude mean and std
    ax = axes[0, 0]
    ax.fill_between(freqs, amp_mean - amp_std, amp_mean + amp_std, alpha=0.3, label="±1σ")
    ax.plot(freqs, amp_mean, "k-", lw=1.5, label="Mean")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log₁₀(Amplitude)")
    ax.set_xscale("log")
    ax.set_title("Amplitude: Mean ± Std")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Amplitude range
    ax = axes[0, 1]
    ax.fill_between(freqs, amp_min, amp_max, alpha=0.3, label="Min-Max")
    ax.plot(freqs, amp_mean, "k-", lw=1.5, label="Mean")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log₁₀(Amplitude)")
    ax.set_xscale("log")
    ax.set_title("Amplitude: Full Range")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phase mean and std
    ax = axes[1, 0]
    ax.fill_between(freqs, phase_mean - phase_std, phase_mean + phase_std, alpha=0.3, label="±1σ")
    ax.plot(freqs, phase_mean, "k-", lw=1.5, label="Mean")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (rad)")
    ax.set_xscale("log")
    ax.set_title("Phase: Mean ± Std")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phase std only (shows variation)
    ax = axes[1, 1]
    ax.plot(freqs, phase_std, "C0-", lw=1.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase Std (rad)")
    ax.set_xscale("log")
    ax.set_title("Phase Variation Across Parameter Space")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/amplitude_phase_stats.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/amplitude_phase_stats.png")


def plot_extreme_cases(f, output_dir="figures/data_check"):
    """Plot waveforms at extreme parameter values."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    freqs = f["frequency_grid"][:]
    params = f["train/parameters"][:]
    log_amp = f["train/log_amplitude"][:]
    phase = f["train/phase"][:]

    # Find extremes
    extremes = {
        "Low η (high q)": np.argmin(params[:, 0]),
        "High η (q~1)": np.argmax(params[:, 0]),
        "High χ₁z": np.argmax(params[:, 1]),
        "Low χ₁z": np.argmin(params[:, 1]),
        "Both spins +": np.argmax(params[:, 1] + params[:, 2]),
        "Both spins -": np.argmin(params[:, 1] + params[:, 2]),
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for name, idx in extremes.items():
        eta, chi1z, chi2z = params[idx]
        label = f"{name}: η={eta:.3f}, χ₁={chi1z:.2f}, χ₂={chi2z:.2f}"
        axes[0].plot(freqs, log_amp[idx], lw=1.2, label=label)
        axes[1].plot(freqs, phase[idx], lw=1.2)

    axes[0].set_ylabel("log₁₀(Amplitude)")
    axes[0].set_title("Extreme Cases in Parameter Space")
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/extreme_cases.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/extreme_cases.png")


def main():
    parser = argparse.ArgumentParser(description="Plot training data for verification")
    parser.add_argument("--input", type=str, default="data/waveforms_22mode.h5",
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
        print(f"  Frequency points: {f['frequency_grid'].shape[0]}")

        plot_sample_waveforms(f, n_samples=args.n_samples, output_dir=args.output_dir)
        plot_parameter_distribution(f, output_dir=args.output_dir)
        plot_amplitude_phase_stats(f, output_dir=args.output_dir)
        plot_extreme_cases(f, output_dir=args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
