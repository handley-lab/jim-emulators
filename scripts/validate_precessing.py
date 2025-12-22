#!/usr/bin/env python
"""
Validate the PRECESSING gravitational waveform emulator.

Usage:
    python scripts/validate_precessing.py
    python scripts/validate_precessing.py --n_test 200
"""

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.waveforms import WaveformParameters, generate_fd_waveform
from jim_emulators.emulator import WaveformPCA, EmulatorMLP


# =============================================================================
# Emulator Loading
# =============================================================================

class PrecessingEmulator:
    """Wrapper for precessing emulator."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.n_hidden = data["n_hidden"]
        self.n_units = data["n_units"]
        self.params_mean = data["params_mean"]
        self.params_std = data["params_std"]
        self.param_names = data["param_names"]
        self.n_params = data["n_params"]
        self.freqs = data["freqs"]
        self.f_min = data["f_min"]
        self.f_max = data["f_max"]
        self.delta_f = data["delta_f"]
        self.M_ref = data["M_ref"]
        self.modes = data["modes"]

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

    def predict_amplitude(self, params: np.ndarray, mode: str) -> np.ndarray:
        """
        Predict log amplitude for a mode.

        Parameters
        ----------
        params : ndarray, shape (7,)
            [eta, chi1, chi2, tilt1, tilt2, phi12, phi_jl]
        mode : str
            Mode string like '22'
        """
        params = np.atleast_2d(params)
        params_norm = (params - self.params_mean) / self.params_std
        params_jax = jnp.array(params_norm)

        md = self.mode_data[mode]
        coeffs_norm = md["model"].apply(md["params"], params_jax)
        coeffs = coeffs_norm * md["target_std"] + md["target_mean"]
        log_amp = md["pca"].inverse_transform(np.array(coeffs))

        return log_amp[0]


# =============================================================================
# Ground Truth
# =============================================================================

def eta_to_masses(eta: float, M_total: float) -> Tuple[float, float]:
    sqrt_term = np.sqrt(1 - 4 * eta)
    q = (1 - sqrt_term) / (1 + sqrt_term)
    return M_total / (1 + q), M_total * q / (1 + q)


def spherical_to_cartesian_spin(chi: float, tilt: float, phi: float) -> Tuple[float, float, float]:
    chi_x = chi * np.sin(tilt) * np.cos(phi)
    chi_y = chi * np.sin(tilt) * np.sin(phi)
    chi_z = chi * np.cos(tilt)
    return chi_x, chi_y, chi_z


def generate_lal_amplitude(
    eta: float, chi1: float, chi2: float,
    tilt1: float, tilt2: float, phi12: float, phi_jl: float,
    l: int, m: int,
    f_min: float, f_max: float, delta_f: float, M_ref: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ground truth amplitude from LAL."""
    m1, m2 = eta_to_masses(eta, M_ref)

    chi1x, chi1y, chi1z = spherical_to_cartesian_spin(chi1, tilt1, phi_jl)
    chi2x, chi2y, chi2z = spherical_to_cartesian_spin(chi2, tilt2, phi_jl + phi12)

    params = WaveformParameters(
        mass_1=m1,
        mass_2=m2,
        chi1x=chi1x, chi1y=chi1y, chi1z=chi1z,
        chi2x=chi2x, chi2y=chi2y, chi2z=chi2z,
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

    valid_mask = freqs >= f_min
    freqs = freqs[valid_mask]
    hp = hp[valid_mask]

    log_amp = np.log10(np.maximum(np.abs(hp), 1e-30))
    return freqs, log_amp


def mode_str_to_lm(mode_str: str) -> Tuple[int, int]:
    return int(mode_str[0]), int(mode_str[1])


# =============================================================================
# Validation
# =============================================================================

def validate_emulator(
    emulator_path: str,
    n_test: int = 100,
    output_dir: str = "figures/validation_precessing",
    seed: int = 123,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Validating PRECESSING Amplitude Emulator")
    print("=" * 60)

    print(f"\nLoading emulator from {emulator_path}...")
    emulator = PrecessingEmulator(emulator_path)
    print(f"  Parameters: {emulator.param_names}")
    print(f"  Modes: {emulator.modes}")
    print(f"  Frequency points: {len(emulator.freqs)}")

    # Sample test parameters
    print(f"\nGenerating {n_test} random test points...")
    rng = np.random.default_rng(seed)
    test_params = np.zeros((n_test, 7))
    test_params[:, 0] = rng.uniform(0.05, 0.25, n_test)      # eta
    test_params[:, 1] = rng.uniform(0.0, 0.99, n_test)       # chi1
    test_params[:, 2] = rng.uniform(0.0, 0.99, n_test)       # chi2
    test_params[:, 3] = rng.uniform(0.0, np.pi, n_test)      # tilt1
    test_params[:, 4] = rng.uniform(0.0, np.pi, n_test)      # tilt2
    test_params[:, 5] = rng.uniform(0.0, 2*np.pi, n_test)    # phi12
    test_params[:, 6] = rng.uniform(0.0, 2*np.pi, n_test)    # phi_jl

    # Validate each mode
    results = {}
    for mode in emulator.modes:
        print(f"\n{'='*40}")
        print(f"Validating Mode {mode}")
        print(f"{'='*40}")

        l, m = mode_str_to_lm(mode)

        mse_list = []
        max_err_list = []

        for i in tqdm(range(n_test), desc=f"Mode {mode}"):
            p = test_params[i]

            # Emulator prediction
            pred = emulator.predict_amplitude(p, mode)

            # LAL ground truth
            _, true = generate_lal_amplitude(
                p[0], p[1], p[2], p[3], p[4], p[5], p[6],
                l, m,
                emulator.f_min, emulator.f_max, emulator.delta_f, emulator.M_ref
            )

            min_len = min(len(pred), len(true))
            pred = pred[:min_len]
            true = true[:min_len]

            mse = float(np.mean((pred - true) ** 2))
            max_err = float(np.max(np.abs(pred - true)))

            mse_list.append(mse)
            max_err_list.append(max_err)

        results[mode] = {
            "mse_mean": np.mean(mse_list),
            "mse_std": np.std(mse_list),
            "max_err_mean": np.mean(max_err_list),
            "max_err_std": np.std(max_err_list),
            "mse_list": mse_list,
            "max_err_list": max_err_list,
        }

        print(f"  MSE: {results[mode]['mse_mean']:.2e} ± {results[mode]['mse_std']:.2e}")
        print(f"  Max Error: {results[mode]['max_err_mean']:.3f} ± {results[mode]['max_err_std']:.3f}")

    # ==========================================================================
    # Plots
    # ==========================================================================
    print("\nGenerating validation plots...")

    # Summary bar plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    modes = list(results.keys())
    x = np.arange(len(modes))

    ax = axes[0]
    mse_means = [results[m]["mse_mean"] for m in modes]
    mse_stds = [results[m]["mse_std"] for m in modes]
    ax.bar(x, mse_means, yerr=mse_stds, alpha=0.7, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("MSE (log amplitude)")
    ax.set_title("Amplitude MSE by Mode (Precessing)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    max_err_means = [results[m]["max_err_mean"] for m in modes]
    max_err_stds = [results[m]["max_err_std"] for m in modes]
    ax.bar(x, max_err_means, yerr=max_err_stds, alpha=0.7, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Max Absolute Error")
    ax.set_title("Max Error by Mode (Precessing)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "validation_summary.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/validation_summary.png")

    # Waveform comparisons
    n_examples = 3
    example_indices = [0, n_test // 2, n_test - 1]

    fig, axes = plt.subplots(len(emulator.modes), n_examples, figsize=(4 * n_examples, 3 * len(emulator.modes)))
    if len(emulator.modes) == 1:
        axes = axes.reshape(1, -1)

    for i, mode in enumerate(emulator.modes):
        l, m = mode_str_to_lm(mode)
        for j, idx in enumerate(example_indices):
            ax = axes[i, j]
            p = test_params[idx]

            pred = emulator.predict_amplitude(p, mode)
            _, true = generate_lal_amplitude(
                p[0], p[1], p[2], p[3], p[4], p[5], p[6],
                l, m,
                emulator.f_min, emulator.f_max, emulator.delta_f, emulator.M_ref
            )

            min_len = min(len(pred), len(true))
            ax.plot(emulator.freqs[:min_len], true[:min_len], 'b-', lw=1.5, label='LAL', alpha=0.8)
            ax.plot(emulator.freqs[:min_len], pred[:min_len], 'r--', lw=1.5, label='Emulator', alpha=0.8)

            ax.set_xscale('log')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(f'log₁₀(A) [{mode}]')
            ax.set_title(f'η={p[0]:.2f}, χ₁={p[1]:.2f}, t₁={p[3]:.2f}')
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "waveform_comparisons.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/waveform_comparisons.png")

    # MSE vs each parameter
    param_names = ['η', 'χ₁', 'χ₂', 'tilt₁', 'tilt₂', 'φ₁₂', 'φ_jl']
    fig, axes = plt.subplots(len(emulator.modes), 7, figsize=(21, 3 * len(emulator.modes)))
    if len(emulator.modes) == 1:
        axes = axes.reshape(1, -1)

    for i, mode in enumerate(emulator.modes):
        mse_arr = np.array(results[mode]['mse_list'])
        for j in range(7):
            ax = axes[i, j]
            ax.scatter(test_params[:, j], mse_arr, alpha=0.5, s=15)
            ax.set_xlabel(param_names[j])
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel(f'MSE [{mode}]')

    axes[0, 3].set_title('MSE vs Parameters (Precessing)')
    plt.tight_layout()
    plt.savefig(output_dir / "mse_vs_params.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/mse_vs_params.png")

    # Special plot: MSE vs precession strength (chi * sin(tilt))
    fig, axes = plt.subplots(1, len(emulator.modes), figsize=(4 * len(emulator.modes), 4))
    if len(emulator.modes) == 1:
        axes = [axes]

    chi_perp_1 = test_params[:, 1] * np.sin(test_params[:, 3])  # chi1 * sin(tilt1)
    chi_perp_2 = test_params[:, 2] * np.sin(test_params[:, 4])  # chi2 * sin(tilt2)
    chi_perp_total = np.sqrt(chi_perp_1**2 + chi_perp_2**2)

    for i, mode in enumerate(emulator.modes):
        ax = axes[i]
        mse_arr = np.array(results[mode]['mse_list'])
        sc = ax.scatter(chi_perp_total, mse_arr, c=test_params[:, 0], cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('χ_perp (precession strength)')
        ax.set_ylabel('MSE')
        ax.set_yscale('log')
        ax.set_title(f'Mode {mode}')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='η')

    plt.suptitle('MSE vs Precession Strength')
    plt.tight_layout()
    plt.savefig(output_dir / "mse_vs_precession.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/mse_vs_precession.png")

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary (Precessing)")
    print("=" * 60)
    for mode in emulator.modes:
        r = results[mode]
        print(f"  Mode {mode}: MSE={r['mse_mean']:.2e}, MaxErr={r['max_err_mean']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate precessing emulator")
    parser.add_argument("--emulator", type=str, default="outputs/emulator_precessing.pkl")
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="figures/validation_precessing")
    parser.add_argument("--seed", type=int, default=123)
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
