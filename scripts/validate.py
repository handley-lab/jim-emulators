#!/usr/bin/env python
"""
Validate the trained waveform emulator.

This script computes comprehensive validation metrics:
1. Waveform reconstruction quality (amplitude and phase errors)
2. Mismatch calculations against LAL reference waveforms
3. Performance across parameter space

Usage:
    python scripts/validate.py                              # Default
    python scripts/validate.py --emulator outputs/emulator_22mode.pkl
    python scripts/validate.py --n_mismatch 500             # More mismatch samples
"""

import argparse
from pathlib import Path
import pickle

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
from jim_emulators.waveforms import WaveformParameters, generate_fd_waveform
from jim_emulators.waveforms.utils import compute_match, load_psd


# =============================================================================
# Emulator Loading and Prediction
# =============================================================================

class DualNetworkEmulator:
    """
    Emulator with separate amplitude and phase networks.
    """

    def __init__(self, emulator_data: dict):
        self.data = emulator_data

        # Load PCA
        self.pca_amplitude = WaveformPCA.from_dict(emulator_data["pca_amplitude"])
        self.pca_phase = WaveformPCA.from_dict(emulator_data["pca_phase"])

        # Load networks
        self.amp_model = EmulatorMLP(
            n_hidden=emulator_data["n_hidden"],
            n_units=emulator_data["n_units"],
            n_outputs=emulator_data["amp_n_outputs"],
        )
        self.phase_model = EmulatorMLP(
            n_hidden=emulator_data["n_hidden"],
            n_units=emulator_data["n_units"],
            n_outputs=emulator_data["phase_n_outputs"],
        )

        # Network parameters
        self.amp_params = emulator_data["amp_params"]
        self.phase_params = emulator_data["phase_params"]

        # Target normalization
        self.amp_target_mean = jnp.array(emulator_data["amp_target_mean"])
        self.amp_target_std = jnp.array(emulator_data["amp_target_std"])
        self.phase_target_mean = jnp.array(emulator_data["phase_target_mean"])
        self.phase_target_std = jnp.array(emulator_data["phase_target_std"])

        # Input normalization
        self.params_mean = jnp.array(emulator_data["params_mean"])
        self.params_std = jnp.array(emulator_data["params_std"])

        # Frequency grid (physical Hz)
        self.frequency_grid = emulator_data["frequency_grid"]

        # Grid parameters for LAL generation
        self.f_min = emulator_data.get("f_min", self.frequency_grid[0])
        self.f_max = emulator_data.get("f_max", self.frequency_grid[-1])
        self.delta_f = emulator_data.get("delta_f", self.frequency_grid[1] - self.frequency_grid[0])
        self.M_ref = emulator_data.get("M_ref", 50.0)

        # JIT compile prediction
        self._predict_amp = jax.jit(self._predict_amp_impl)
        self._predict_phase = jax.jit(self._predict_phase_impl)

    def _predict_amp_impl(self, params_norm):
        pred_norm = self.amp_model.apply(self.amp_params, params_norm)
        return pred_norm * self.amp_target_std + self.amp_target_mean

    def _predict_phase_impl(self, params_norm):
        pred_norm = self.phase_model.apply(self.phase_params, params_norm)
        return pred_norm * self.phase_target_std + self.phase_target_mean

    def predict(self, params: np.ndarray):
        """
        Predict amplitude and phase.

        Parameters
        ----------
        params : ndarray, shape (n_samples, 3) or (3,)
            Parameters [eta, chi1z, chi2z].

        Returns
        -------
        log_amplitude : ndarray, shape (n_samples, n_freq)
        phase : ndarray, shape (n_samples, n_freq)
        """
        params = np.atleast_2d(params)
        params_norm = (params - self.params_mean) / self.params_std
        params_norm = jnp.array(params_norm)

        # Predict PCA coefficients
        amp_coeffs = np.array(self._predict_amp(params_norm))
        phase_coeffs = np.array(self._predict_phase(params_norm))

        # Reconstruct waveforms
        log_amplitude = self.pca_amplitude.inverse_transform(amp_coeffs)
        phase = self.pca_phase.inverse_transform(phase_coeffs)

        return log_amplitude, phase


def load_emulator(path: str) -> DualNetworkEmulator:
    """Load emulator from file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return DualNetworkEmulator(data)


# =============================================================================
# Mismatch Calculation
# =============================================================================

def eta_to_masses(eta: float, M_total: float):
    """Convert symmetric mass ratio to component masses."""
    sqrt_term = np.sqrt(1 - 4 * eta)
    q = (1 - sqrt_term) / (1 + sqrt_term)
    m1 = M_total / (1 + q)
    m2 = M_total * q / (1 + q)
    return m1, m2


def compute_mismatch_lal(
    emulator: DualNetworkEmulator,
    eta: float,
    chi1z: float,
    chi2z: float,
    psd_file: str = "psds/ET-D-psd.txt",
) -> dict:
    """
    Compute mismatch between emulator and LAL waveform.

    Uses the SAME frequency grid as training - no interpolation.
    """
    # Get emulator prediction
    params = np.array([[eta, chi1z, chi2z]])
    log_amp_emu, phase_emu = emulator.predict(params)
    log_amp_emu = log_amp_emu[0]
    phase_emu = phase_emu[0]

    # Convert to complex strain
    amp_emu = 10.0 ** log_amp_emu
    h_emu = amp_emu * np.exp(-1j * phase_emu)

    # Generate LAL waveform with EXACT same grid parameters
    m1, m2 = eta_to_masses(eta, emulator.M_ref)

    lal_params = WaveformParameters(
        mass_1=m1,
        mass_2=m2,
        chi1z=chi1z,
        chi2z=chi2z,
        luminosity_distance=1.0,
        inclination=0.0,
        phase=0.0,
        f_min=emulator.f_min,
        f_max=emulator.f_max,
        delta_f=emulator.delta_f,
        approximant="IMRPhenomXPHM",
    )

    # Generate LAL waveform - same grid, no interpolation needed
    freqs_lal, hp_lal, _ = generate_fd_waveform(
        lal_params, mode_array=[(2, 2), (2, -2)], disable_multibanding=True
    )

    # Extract amplitude and phase directly
    amp_lal = np.abs(hp_lal)
    phase_lal = np.unwrap(np.angle(hp_lal))

    # Align phase at peak amplitude
    peak_idx = np.argmax(amp_lal)
    phase_lal = phase_lal - phase_lal[peak_idx]

    # Reconstruct aligned LAL strain
    h_lal = amp_lal * np.exp(-1j * phase_lal)

    # Also align emulator phase at same peak location for comparison
    phase_emu_aligned = phase_emu - phase_emu[peak_idx]
    h_emu_aligned = amp_emu * np.exp(-1j * phase_emu_aligned)

    # Load and interpolate PSD to frequency grid
    psd_freqs, psd_values = load_psd(psd_file)
    psd_interp = np.interp(emulator.frequency_grid, np.array(psd_freqs), np.array(psd_values))

    # Compute match
    match = float(compute_match(
        jnp.array(h_emu_aligned),
        jnp.array(h_lal),
        jnp.array(psd_interp),
        jnp.array(emulator.frequency_grid)
    ))
    mismatch = 1.0 - match

    return {
        "mismatch": mismatch,
        "match": match,
        "eta": eta,
        "chi1z": chi1z,
        "chi2z": chi2z,
        "amp_lal": amp_lal,
        "amp_emu": amp_emu,
        "phase_lal": phase_lal,
        "phase_emu": phase_emu_aligned,
        "log_amp_lal": np.log10(np.maximum(amp_lal, 1e-100)),
        "log_amp_emu": log_amp_emu,
    }


# =============================================================================
# Validation
# =============================================================================

def validate_emulator(
    emulator_path: str,
    data_path: str,
    output_dir: str = "outputs/validation",
    n_mismatch_samples: int = 200,
    psd_file: str = "psds/ET-D-psd.txt",
):
    """Comprehensive validation of trained emulator."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Validating Waveform Emulator (Dual Network)")
    print("=" * 60)

    # Load emulator
    print(f"\nLoading emulator from {emulator_path}...")
    emulator = load_emulator(emulator_path)
    print(f"  Frequency grid: {len(emulator.frequency_grid)} points")
    print(f"  Frequency range: {emulator.f_min:.1f} - {emulator.f_max:.1f} Hz")
    print(f"  Amplitude PCA: {emulator.pca_amplitude.n_components} components")
    print(f"  Phase PCA: {emulator.pca_phase.n_components} components")

    # Load validation data
    print(f"\nLoading validation data from {data_path}...")
    with h5py.File(data_path, "r") as f:
        val_params = f["validation/parameters"][:]
        val_log_amp = f["validation/log_amplitude"][:]
        val_phase = f["validation/phase"][:]
        freq_grid = f["frequency_grid"][:]

    print(f"  Validation samples: {len(val_params)}")

    # ==========================================================================
    # 1. Reconstruction Quality
    # ==========================================================================
    print("\n[1/3] Evaluating reconstruction quality...")

    pred_log_amp, pred_phase = emulator.predict(val_params)

    # Amplitude errors
    amp_residuals = pred_log_amp - val_log_amp
    amp_mse = np.mean(amp_residuals ** 2)
    amp_max_error = np.max(np.abs(amp_residuals))

    print(f"  Amplitude (log10):")
    print(f"    MSE: {amp_mse:.2e}")
    print(f"    Max error: {amp_max_error:.4f}")

    # Phase errors
    phase_residuals = pred_phase - val_phase
    phase_mse = np.mean(phase_residuals ** 2)
    phase_max_error = np.max(np.abs(phase_residuals))

    print(f"  Phase (rad):")
    print(f"    MSE: {phase_mse:.2e}")
    print(f"    Max error: {phase_max_error:.4f} rad")

    # ==========================================================================
    # 2. Mismatch Against LAL
    # ==========================================================================
    print(f"\n[2/3] Computing mismatches ({n_mismatch_samples} samples)...")

    rng = np.random.default_rng(42)
    mismatch_indices = rng.choice(
        len(val_params),
        size=min(n_mismatch_samples, len(val_params)),
        replace=False
    )

    mismatches = []
    mismatch_params = []
    failed = 0

    for idx in tqdm(mismatch_indices, desc="Mismatch"):
        eta, chi1z, chi2z = val_params[idx]
        try:
            result = compute_mismatch_lal(emulator, eta, chi1z, chi2z, psd_file=psd_file)
            mismatches.append(result["mismatch"])
            mismatch_params.append([eta, chi1z, chi2z])
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"\n  Warning: {e}")

    mismatches = np.array(mismatches)
    mismatch_params = np.array(mismatch_params)

    print(f"\n  Mismatch Statistics ({len(mismatches)} successful):")
    print(f"    Median: {np.median(mismatches):.2e}")
    print(f"    Mean: {np.mean(mismatches):.2e}")
    print(f"    Max: {np.max(mismatches):.2e}")
    print(f"    Min: {np.min(mismatches):.2e}")
    print(f"    < 1e-3: {np.sum(mismatches < 1e-3) / len(mismatches) * 100:.1f}%")
    print(f"    < 1e-2: {np.sum(mismatches < 1e-2) / len(mismatches) * 100:.1f}%")

    # ==========================================================================
    # 3. Generate Plots
    # ==========================================================================
    print("\n[3/3] Generating plots...")

    # --- Reconstruction quality ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    ax.hist(amp_residuals.flatten(), bins=100, density=True, alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('log₁₀(Amplitude) Residual')
    ax.set_title(f'Amplitude Residuals (MSE={amp_mse:.2e})')

    ax = axes[0, 1]
    ax.hist(phase_residuals.flatten(), bins=100, density=True, alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Phase Residual (rad)')
    ax.set_title(f'Phase Residuals (MSE={phase_mse:.2e})')

    ax = axes[0, 2]
    ax.semilogy(freq_grid, np.mean(amp_residuals**2, axis=0), label='Amplitude')
    ax.semilogy(freq_grid, np.mean(phase_residuals**2, axis=0), label='Phase')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xscale('log')
    ax.set_title('Per-Frequency MSE')
    ax.legend()

    for i, idx in enumerate([0, len(val_params)//2, len(val_params)-1]):
        ax = axes[1, i]
        ax.plot(freq_grid, val_log_amp[idx], 'b-', label='True', lw=2)
        ax.plot(freq_grid, pred_log_amp[idx], 'r--', label='Emulator', lw=2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_xscale('log')
        eta, chi1z, chi2z = val_params[idx]
        ax.set_title(f'η={eta:.3f}, χ₁={chi1z:.2f}, χ₂={chi2z:.2f}')
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_quality.png", dpi=150)
    plt.close()
    print(f"  Saved: reconstruction_quality.png")

    # --- Mismatch analysis ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.hist(np.log10(mismatches), bins=50, alpha=0.7)
    ax.axvline(np.log10(1e-3), color='g', linestyle='--', label='Target')
    ax.axvline(np.log10(np.median(mismatches)), color='r', label=f'Median')
    ax.set_xlabel('log₁₀(Mismatch)')
    ax.set_title('Mismatch Distribution')
    ax.legend()

    ax = axes[0, 1]
    sc = ax.scatter(mismatch_params[:, 0], mismatches, c=mismatch_params[:, 1], cmap='coolwarm', s=20)
    ax.axhline(1e-3, color='g', linestyle='--')
    ax.set_xlabel('η')
    ax.set_ylabel('Mismatch')
    ax.set_yscale('log')
    ax.set_title('Mismatch vs η')
    plt.colorbar(sc, ax=ax, label='χ₁z')

    ax = axes[1, 0]
    sc = ax.scatter(mismatch_params[:, 1], mismatches, c=mismatch_params[:, 0], cmap='viridis', s=20)
    ax.axhline(1e-3, color='g', linestyle='--')
    ax.set_xlabel('χ₁z')
    ax.set_ylabel('Mismatch')
    ax.set_yscale('log')
    ax.set_title('Mismatch vs χ₁z')
    plt.colorbar(sc, ax=ax, label='η')

    ax = axes[1, 1]
    sc = ax.scatter(mismatch_params[:, 1], mismatch_params[:, 2], c=np.log10(mismatches), cmap='RdYlGn_r', s=20)
    ax.set_xlabel('χ₁z')
    ax.set_ylabel('χ₂z')
    ax.set_title('Mismatch in Spin Space')
    plt.colorbar(sc, ax=ax, label='log₁₀(Mismatch)')

    plt.tight_layout()
    plt.savefig(output_dir / "mismatch_analysis.png", dpi=150)
    plt.close()
    print(f"  Saved: mismatch_analysis.png")

    # --- Waveform comparisons ---
    worst_idx = np.argmax(mismatches)
    best_idx = np.argmin(mismatches)
    median_idx = np.argsort(mismatches)[len(mismatches)//2]

    cases = [
        ("Best", best_idx, mismatches[best_idx]),
        ("Median", median_idx, mismatches[median_idx]),
        ("Worst", worst_idx, mismatches[worst_idx]),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for row, (label, idx, mm) in enumerate(cases):
        eta, chi1z, chi2z = mismatch_params[idx]
        result = compute_mismatch_lal(emulator, eta, chi1z, chi2z, psd_file=psd_file)

        ax = axes[row, 0]
        ax.plot(freq_grid, result["log_amp_lal"], 'b-', label='LAL', lw=2)
        ax.plot(freq_grid, result["log_amp_emu"], 'r--', label='Emulator', lw=2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('log₁₀(Amplitude)')
        ax.set_xscale('log')
        ax.set_title(f'{label}: η={eta:.3f}, χ₁={chi1z:.2f}, χ₂={chi2z:.2f}\nMismatch={mm:.2e}')
        ax.legend()

        ax = axes[row, 1]
        ax.plot(freq_grid, result["phase_lal"], 'b-', label='LAL', lw=2)
        ax.plot(freq_grid, result["phase_emu"], 'r--', label='Emulator', lw=2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase (rad)')
        ax.set_xscale('log')
        ax.set_title('Phase Comparison')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "waveform_comparisons.png", dpi=150)
    plt.close()
    print(f"  Saved: waveform_comparisons.png")

    # --- Summary ---
    fig, ax = plt.subplots(figsize=(8, 6))
    stats_text = f"""
Emulator Validation Summary
===========================

Reconstruction (Validation Set)
  Amplitude MSE: {amp_mse:.2e}
  Phase MSE: {phase_mse:.2e}
  Amplitude max error: {amp_max_error:.4f}
  Phase max error: {phase_max_error:.4f} rad

Mismatch Statistics (n={len(mismatches)})
  Median: {np.median(mismatches):.2e}
  Mean: {np.mean(mismatches):.2e}
  Max: {np.max(mismatches):.2e}
  Min: {np.min(mismatches):.2e}

  Fraction < 10⁻³: {np.sum(mismatches < 1e-3) / len(mismatches) * 100:.1f}%
  Fraction < 10⁻²: {np.sum(mismatches < 1e-2) / len(mismatches) * 100:.1f}%

Architecture
  Amplitude PCA: {emulator.pca_amplitude.n_components} components
  Phase PCA: {emulator.pca_phase.n_components} components
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    plt.savefig(output_dir / "validation_summary.png", dpi=150)
    plt.close()
    print(f"  Saved: validation_summary.png")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate emulator")
    parser.add_argument("--emulator", type=str, default="outputs/emulator_22mode.pkl")
    parser.add_argument("--data", type=str, default="data/waveforms_22mode.h5")
    parser.add_argument("--output", type=str, default="outputs/validation")
    parser.add_argument("--n_mismatch", type=int, default=200)
    parser.add_argument("--psd", type=str, default="psds/ET-D-psd.txt")
    args = parser.parse_args()

    validate_emulator(
        emulator_path=args.emulator,
        data_path=args.data,
        output_dir=args.output,
        n_mismatch_samples=args.n_mismatch,
        psd_file=args.psd,
    )


if __name__ == "__main__":
    main()
