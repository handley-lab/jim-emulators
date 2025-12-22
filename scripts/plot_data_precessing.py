#!/usr/bin/env python
"""
Plot precessing training data to verify waveform generation quality.

Usage:
    python scripts/plot_data_precessing.py
    python scripts/plot_data_precessing.py --input data/waveforms_precessing_test.h5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def plot_sample_waveforms(f, n_samples=10, output_dir="figures/data_check_precessing"):
    """Plot sample waveforms for all modes."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    freqs = f["freqs"][:]
    params = f["train/parameters"][:]
    modes = list(f.attrs["modes"])
    param_names = list(f.attrs["param_names"])

    n_total = len(params)
    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)

    # Color by precession strength
    chi_perp = params[indices, 1] * np.sin(params[indices, 3])  # chi1 * sin(tilt1)
    colors = plt.cm.plasma((chi_perp - chi_perp.min()) / (chi_perp.max() - chi_perp.min() + 1e-10))

    n_modes = len(modes)
    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 3 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        log_amp = f[f"train/log_amplitude_{mode}"][:]
        for i, idx in enumerate(indices):
            ax.plot(freqs, log_amp[idx], color=colors[i], alpha=0.7, lw=0.8)
        ax.set_ylabel(f"log₁₀(A) [{mode}]")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"Sample Precessing Waveforms (n={n_samples}, colored by χ⊥)")
    axes[-1].set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_waveforms.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/sample_waveforms.png")


def plot_parameter_distributions(f, output_dir="figures/data_check_precessing"):
    """Plot 1D histograms for all parameters."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    params_train = f["train/parameters"][:]
    params_val = f["validation/parameters"][:]
    param_names = list(f.attrs["param_names"])

    n_params = len(param_names)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    labels = ["η", "χ₁", "χ₂", "tilt₁", "tilt₂", "φ₁₂", "φ_jl"]

    for i in range(n_params):
        ax = axes[i]
        ax.hist(params_train[:, i], bins=50, alpha=0.7, label="Train", density=True)
        ax.hist(params_val[:, i], bins=30, alpha=0.7, label="Validation", density=True)
        ax.set_xlabel(labels[i])
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[-1].axis("off")

    plt.suptitle("Parameter Space Coverage (7D Precessing)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_distributions.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/parameter_distributions.png")


def plot_parameter_correlations(f, output_dir="figures/data_check_precessing"):
    """Plot key 2D parameter correlations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    params = f["train/parameters"][:]

    # Key pairs to visualize
    pairs = [
        (0, 1, "η", "χ₁"),
        (1, 3, "χ₁", "tilt₁"),
        (2, 4, "χ₂", "tilt₂"),
        (3, 4, "tilt₁", "tilt₂"),
        (1, 2, "χ₁", "χ₂"),
        (5, 6, "φ₁₂", "φ_jl"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (i, j, xlabel, ylabel) in zip(axes, pairs):
        ax.scatter(params[:, i], params[:, j], alpha=0.1, s=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Parameter Correlations (LHS Sampling)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_correlations.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/parameter_correlations.png")


def plot_amplitude_stats(f, output_dir="figures/data_check_precessing"):
    """Plot amplitude statistics per mode."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    freqs = f["freqs"][:]
    modes = list(f.attrs["modes"])

    n_modes = len(modes)
    fig, axes = plt.subplots(n_modes, 2, figsize=(12, 3 * n_modes))
    if n_modes == 1:
        axes = axes.reshape(1, -1)

    for i, mode in enumerate(modes):
        log_amp = f[f"train/log_amplitude_{mode}"][:]

        amp_mean = np.mean(log_amp, axis=0)
        amp_std = np.std(log_amp, axis=0)
        amp_min = np.min(log_amp, axis=0)
        amp_max = np.max(log_amp, axis=0)

        ax = axes[i, 0]
        ax.fill_between(freqs, amp_mean - amp_std, amp_mean + amp_std, alpha=0.3, label="±1σ")
        ax.plot(freqs, amp_mean, "k-", lw=1.5, label="Mean")
        ax.set_ylabel(f"log₁₀(A) [{mode}]")
        ax.set_xscale("log")
        ax.set_title(f"Mode {mode}: Mean ± Std")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.fill_between(freqs, amp_min, amp_max, alpha=0.3, label="Min-Max")
        ax.plot(freqs, amp_mean, "k-", lw=1.5, label="Mean")
        ax.set_ylabel(f"log₁₀(A) [{mode}]")
        ax.set_xscale("log")
        ax.set_title(f"Mode {mode}: Full Range")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Frequency (Hz)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/amplitude_stats.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/amplitude_stats.png")


def plot_precession_dependence(f, output_dir="figures/data_check_precessing"):
    """Plot how waveforms depend on precession strength."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    freqs = f["freqs"][:]
    params = f["train/parameters"][:]
    modes = list(f.attrs["modes"])

    # Compute precession strength: chi_perp = sqrt((chi1*sin(tilt1))^2 + (chi2*sin(tilt2))^2)
    chi_perp_1 = params[:, 1] * np.sin(params[:, 3])
    chi_perp_2 = params[:, 2] * np.sin(params[:, 4])
    chi_perp = np.sqrt(chi_perp_1**2 + chi_perp_2**2)

    # Bin by precession strength
    bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    bin_labels = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0"]

    n_modes = len(modes)
    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 3 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        log_amp = f[f"train/log_amplitude_{mode}"][:]

        for j in range(len(bins) - 1):
            mask = (chi_perp >= bins[j]) & (chi_perp < bins[j+1])
            if np.sum(mask) > 0:
                mean_amp = np.mean(log_amp[mask], axis=0)
                ax.plot(freqs, mean_amp, lw=1.5, label=f"χ⊥={bin_labels[j]}", alpha=0.8)

        ax.set_ylabel(f"log₁₀(A) [{mode}]")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_title("Mean Amplitude by Precession Strength (χ⊥)")
    axes[-1].set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/precession_dependence.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/precession_dependence.png")


def plot_mode_comparison(f, output_dir="figures/data_check_precessing"):
    """Compare modes for different precession strengths."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    freqs = f["freqs"][:]
    params = f["train/parameters"][:]
    modes = list(f.attrs["modes"])

    # Compute precession strength
    chi_perp_1 = params[:, 1] * np.sin(params[:, 3])
    chi_perp_2 = params[:, 2] * np.sin(params[:, 4])
    chi_perp = np.sqrt(chi_perp_1**2 + chi_perp_2**2)

    # Find samples at different precession levels
    sorted_idx = np.argsort(chi_perp)
    indices = [
        sorted_idx[0],                    # Low precession
        sorted_idx[len(sorted_idx)//2],   # Medium precession
        sorted_idx[-1],                   # High precession
    ]
    titles = ["Low χ⊥", "Medium χ⊥", "High χ⊥"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, idx, title in zip(axes, indices, titles):
        p = params[idx]
        chi_p = chi_perp[idx]

        for mode in modes:
            log_amp = f[f"train/log_amplitude_{mode}"][idx]
            ax.plot(freqs, log_amp, label=mode, lw=1.2)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("log₁₀(Amplitude)")
        ax.set_xscale("log")
        ax.set_title(f"{title} (χ⊥={chi_p:.2f})\nη={p[0]:.2f}, χ₁={p[1]:.2f}, t₁={p[3]:.1f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Mode Comparison at Different Precession Strengths")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mode_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/mode_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Plot precessing training data")
    parser.add_argument("--input", type=str, default="data/waveforms_precessing.h5")
    parser.add_argument("--output_dir", type=str, default="figures/data_check_precessing")
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    with h5py.File(args.input, "r") as f:
        print(f"  Train samples: {f['train/parameters'].shape[0]}")
        print(f"  Validation samples: {f['validation/parameters'].shape[0]}")
        print(f"  Parameters: {list(f.attrs['param_names'])}")
        print(f"  Frequency points: {f['freqs'].shape[0]}")
        print(f"  Modes: {list(f.attrs['modes'])}")

        plot_sample_waveforms(f, n_samples=args.n_samples, output_dir=args.output_dir)
        plot_parameter_distributions(f, output_dir=args.output_dir)
        plot_parameter_correlations(f, output_dir=args.output_dir)
        plot_amplitude_stats(f, output_dir=args.output_dir)
        plot_precession_dependence(f, output_dir=args.output_dir)
        plot_mode_comparison(f, output_dir=args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
