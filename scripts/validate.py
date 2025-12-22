#!/usr/bin/env python
"""
Validate the trained gravitational waveform emulator.

Compares emulator predictions against LAL ground truth for amplitude.

Usage:
    python scripts/validate.py                              # Default
    python scripts/validate.py --emulator outputs/emulator_multimode.pkl
    python scripts/validate.py --n_test 100                 # More test points
"""

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.waveforms import WaveformParameters, generate_fd_waveform
from jim_emulators.waveforms.utils import geometric_to_physical_frequency
from jim_emulators.emulator import WaveformPCA, EmulatorMLP


# =============================================================================
# Emulator Loading
# =============================================================================

class MultimodeEmulator:
    """Wrapper for loading and using the trained multimode emulator."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.n_hidden = data["n_hidden"]
        self.n_units = data["n_units"]
        self.params_mean = data["params_mean"]
        self.params_std = data["params_std"]
        self.freqs = data["freqs"]
        self.f_min = data["f_min"]
        self.f_max = data["f_max"]
        self.delta_f = data["delta_f"]
        self.M_ref = data["M_ref"]
        self.modes = data["modes"]

        # Load per-mode networks and PCA
        self.mode_data = {}
        for mode in self.modes:
            mode_info = data[f"mode_{mode}"]
            pca = WaveformPCA.from_dict(mode_info["pca"])
            model = EmulatorMLP(
                n_hidden=self.n_hidden,
                n_units=self.n_units,
                n_outputs=mode_info["n_outputs"],
            )
            self.mode_data[mode] = {
                "model": model,
                "params": mode_info["params"],
                "target_mean": mode_info["target_mean"],
                "target_std": mode_info["target_std"],
                "pca": pca,
            }

    def predict_amplitude(self, eta: float, chi1z: float, chi2z: float, mode: str) -> np.ndarray:
        """Predict log amplitude for a single mode."""
        # Normalize inputs
        params = np.array([[eta, chi1z, chi2z]])
        params_norm = (params - self.params_mean) / self.params_std
        params_jax = jnp.array(params_norm)

        # Get mode data
        md = self.mode_data[mode]

        # Predict PCA coefficients
        coeffs_norm = md["model"].apply(md["params"], params_jax)
        coeffs = coeffs_norm * md["target_std"] + md["target_mean"]

        # Inverse PCA to get amplitude
        log_amp = md["pca"].inverse_transform(np.array(coeffs))

        return log_amp[0]


# =============================================================================
# Ground Truth Generation
# =============================================================================

def eta_to_masses(eta: float, M_total: float) -> Tuple[float, float]:
    sqrt_term = np.sqrt(1 - 4 * eta)
    q = (1 - sqrt_term) / (1 + sqrt_term)
    return M_total / (1 + q), M_total * q / (1 + q)


def generate_lal_amplitude(
    eta: float, chi1z: float, chi2z: float,
    l: int, m: int,
    f_min: float, f_max: float, delta_f: float, M_ref: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ground truth amplitude from LAL."""
    m1, m2 = eta_to_masses(eta, M_ref)

    params = WaveformParameters(
        mass_1=m1,
        mass_2=m2,
        chi1z=chi1z,
        chi2z=chi2z,
        luminosity_distance=1.0,
        inclination=0.0,
        phase=0.0,
        f_min=f_min,
        f_max=f_max,
        delta_f=delta_f,
        approximant="IMRPhenomXPHM",
    )

    mode_array = [(l, m), (l, -m)]
    freqs, hp, _ = generate_fd_waveform(params, mode_array=mode_array, disable_multibanding=True)

    # Slice to f >= f_min
    valid_mask = freqs >= f_min
    freqs = freqs[valid_mask]
    hp = hp[valid_mask]

    # Log amplitude with -30 floor
    log_amp = np.log10(np.maximum(np.abs(hp), 1e-30))

    return freqs, log_amp


def mode_str_to_lm(mode_str: str) -> Tuple[int, int]:
    """Convert '22' to (2, 2)."""
    return int(mode_str[0]), int(mode_str[1])


# =============================================================================
# Validation Metrics
# =============================================================================

def compute_amplitude_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared error in log amplitude."""
    return float(np.mean((pred - true) ** 2))


def compute_amplitude_max_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Maximum absolute error in log amplitude."""
    return float(np.max(np.abs(pred - true)))


def compute_relative_amplitude_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Relative error in linear amplitude: |10^pred - 10^true| / 10^true."""
    amp_pred = 10 ** pred
    amp_true = 10 ** true
    # Avoid division by very small numbers
    amp_true_safe = np.maximum(amp_true, 1e-30)
    return np.abs(amp_pred - amp_true) / amp_true_safe


# =============================================================================
# Main Validation
# =============================================================================

def validate_emulator(
    emulator_path: str,
    n_test: int = 50,
    output_dir: str = "figures/validation",
    seed: int = 123,
):
    """Run full validation of the emulator."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Validating Multimode Amplitude Emulator")
    print("=" * 60)

    # Load emulator
    print(f"\nLoading emulator from {emulator_path}...")
    emulator = MultimodeEmulator(emulator_path)
    print(f"  Modes: {emulator.modes}")
    print(f"  Frequency points: {len(emulator.freqs)}")

    # Generate random test parameters
    print(f"\nGenerating {n_test} random test points...")
    rng = np.random.default_rng(seed)
    test_params = np.zeros((n_test, 3))
    test_params[:, 0] = rng.uniform(0.05, 0.25, n_test)  # eta
    test_params[:, 1] = rng.uniform(-0.99, 0.99, n_test)  # chi1z
    test_params[:, 2] = rng.uniform(-0.99, 0.99, n_test)  # chi2z

    # Validate each mode
    results = {}
    for mode in emulator.modes:
        print(f"\n{'='*40}")
        print(f"Validating Mode {mode}")
        print(f"{'='*40}")

        l, m = mode_str_to_lm(mode)

        mse_list = []
        max_err_list = []
        rel_err_list = []

        for i in tqdm(range(n_test), desc=f"Mode {mode}"):
            eta, chi1z, chi2z = test_params[i]

            # Emulator prediction
            pred = emulator.predict_amplitude(eta, chi1z, chi2z, mode)

            # LAL ground truth
            _, true = generate_lal_amplitude(
                eta, chi1z, chi2z, l, m,
                emulator.f_min, emulator.f_max, emulator.delta_f, emulator.M_ref
            )

            # Ensure same length (may differ by 1-2 points due to grid)
            min_len = min(len(pred), len(true))
            pred = pred[:min_len]
            true = true[:min_len]

            mse = compute_amplitude_mse(pred, true)
            max_err = compute_amplitude_max_error(pred, true)
            rel_err = compute_relative_amplitude_error(pred, true)

            mse_list.append(mse)
            max_err_list.append(max_err)
            rel_err_list.append(np.median(rel_err))

        results[mode] = {
            "mse_mean": np.mean(mse_list),
            "mse_std": np.std(mse_list),
            "max_err_mean": np.mean(max_err_list),
            "max_err_std": np.std(max_err_list),
            "rel_err_median": np.median(rel_err_list),
            "mse_list": mse_list,
            "max_err_list": max_err_list,
        }

        print(f"  MSE: {results[mode]['mse_mean']:.2e} ± {results[mode]['mse_std']:.2e}")
        print(f"  Max Error: {results[mode]['max_err_mean']:.3f} ± {results[mode]['max_err_std']:.3f}")
        print(f"  Median Rel Error: {results[mode]['rel_err_median']:.2e}")

    # ==========================================================================
    # Summary Plots
    # ==========================================================================
    print("\nGenerating validation plots...")

    # Plot 1: MSE distribution per mode
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    modes = list(results.keys())
    mse_means = [results[m]["mse_mean"] for m in modes]
    mse_stds = [results[m]["mse_std"] for m in modes]
    x = np.arange(len(modes))
    ax.bar(x, mse_means, yerr=mse_stds, alpha=0.7, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("MSE (log amplitude)")
    ax.set_title("Amplitude MSE by Mode")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    max_err_means = [results[m]["max_err_mean"] for m in modes]
    max_err_stds = [results[m]["max_err_std"] for m in modes]
    ax.bar(x, max_err_means, yerr=max_err_stds, alpha=0.7, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Max Absolute Error (log amplitude)")
    ax.set_title("Max Error by Mode")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "validation_summary.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/validation_summary.png")

    # Plot 2: Example waveform comparisons
    n_examples = min(3, n_test)
    fig, axes = plt.subplots(len(emulator.modes), n_examples, figsize=(4 * n_examples, 3 * len(emulator.modes)))
    if len(emulator.modes) == 1:
        axes = axes.reshape(1, -1)

    example_indices = [0, n_test // 2, n_test - 1][:n_examples]

    for i, mode in enumerate(emulator.modes):
        l, m = mode_str_to_lm(mode)
        for j, idx in enumerate(example_indices):
            ax = axes[i, j]
            eta, chi1z, chi2z = test_params[idx]

            pred = emulator.predict_amplitude(eta, chi1z, chi2z, mode)
            freqs_true, true = generate_lal_amplitude(
                eta, chi1z, chi2z, l, m,
                emulator.f_min, emulator.f_max, emulator.delta_f, emulator.M_ref
            )

            min_len = min(len(pred), len(true))
            ax.plot(emulator.freqs[:min_len], true[:min_len], 'b-', lw=1.5, label='LAL', alpha=0.8)
            ax.plot(emulator.freqs[:min_len], pred[:min_len], 'r--', lw=1.5, label='Emulator', alpha=0.8)

            ax.set_xscale('log')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(f'log₁₀(A) [{mode}]')
            ax.set_title(f'η={eta:.3f}, χ₁={chi1z:.2f}, χ₂={chi2z:.2f}')
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "waveform_comparisons.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/waveform_comparisons.png")

    # Plot 3: Residuals for one example
    fig, axes = plt.subplots(len(emulator.modes), 1, figsize=(12, 3 * len(emulator.modes)), sharex=True)
    if len(emulator.modes) == 1:
        axes = [axes]

    idx = n_test // 2
    eta, chi1z, chi2z = test_params[idx]

    for i, mode in enumerate(emulator.modes):
        ax = axes[i]
        l, m = mode_str_to_lm(mode)

        pred = emulator.predict_amplitude(eta, chi1z, chi2z, mode)
        _, true = generate_lal_amplitude(
            eta, chi1z, chi2z, l, m,
            emulator.f_min, emulator.f_max, emulator.delta_f, emulator.M_ref
        )

        min_len = min(len(pred), len(true))
        residual = pred[:min_len] - true[:min_len]

        ax.plot(emulator.freqs[:min_len], residual, 'k-', lw=0.8)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.fill_between(emulator.freqs[:min_len], -0.01, 0.01, alpha=0.2, color='green', label='±0.01')
        ax.set_xscale('log')
        ax.set_ylabel(f'Residual [{mode}]')
        ax.set_ylim(-0.1, 0.1)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel('Frequency (Hz)')
    axes[0].set_title(f'Residuals (Emulator - LAL): η={eta:.3f}, χ₁={chi1z:.2f}, χ₂={chi2z:.2f}')

    plt.tight_layout()
    plt.savefig(output_dir / "residuals.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/residuals.png")

    # Plot 4: MSE vs parameter values
    fig, axes = plt.subplots(len(emulator.modes), 3, figsize=(12, 3 * len(emulator.modes)))
    if len(emulator.modes) == 1:
        axes = axes.reshape(1, -1)

    param_names = ['η', 'χ₁z', 'χ₂z']
    for i, mode in enumerate(emulator.modes):
        mse_arr = np.array(results[mode]['mse_list'])
        for j in range(3):
            ax = axes[i, j]
            ax.scatter(test_params[:, j], mse_arr, alpha=0.5, s=20)
            ax.set_xlabel(param_names[j])
            ax.set_ylabel('MSE')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel(f'MSE [{mode}]')

    axes[0, 1].set_title('MSE vs Parameters')
    plt.tight_layout()
    plt.savefig(output_dir / "mse_vs_params.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/mse_vs_params.png")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    for mode in emulator.modes:
        r = results[mode]
        print(f"  Mode {mode}: MSE={r['mse_mean']:.2e}, MaxErr={r['max_err_mean']:.3f}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate waveform emulator")
    parser.add_argument("--emulator", type=str, default="outputs/emulator_multimode.pkl",
                        help="Path to trained emulator")
    parser.add_argument("--n_test", type=int, default=50,
                        help="Number of test points")
    parser.add_argument("--output_dir", type=str, default="figures/validation",
                        help="Output directory for plots")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for test parameters")
    args = parser.parse_args()

    validate_emulator(
        emulator_path=args.emulator,
        n_test=args.n_test,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
