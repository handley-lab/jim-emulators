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
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

import jax
jax.config.update("jax_enable_x64", True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.emulator import GWEmulator
from jim_emulators.waveforms import WaveformParameters, generate_fd_waveform
from jim_emulators.waveforms.utils import (
    geometric_to_physical_frequency,
    physical_to_geometric_frequency,
    compute_match,
    load_psd,
)


# =============================================================================
# Mismatch Calculation
# =============================================================================

def compute_mismatch_lal(
    emulator: GWEmulator,
    eta: float,
    chi1z: float,
    chi2z: float,
    psd_file: str = "psds/ET-D-psd.txt",
    M_total: float = 50.0,
) -> dict:
    """
    Compute mismatch between emulator and LAL waveform.

    Parameters
    ----------
    emulator : GWEmulator
        Trained emulator.
    eta, chi1z, chi2z : float
        Binary parameters.
    psd_file : str
        Path to PSD file.
    M_total : float
        Reference total mass (solar masses).

    Returns
    -------
    result : dict
        Dictionary with mismatch and intermediate values.
    """
    # Get emulator prediction
    params = np.array([[eta, chi1z, chi2z]])
    log_amp_emu, phase_emu = emulator.predict(params)
    log_amp_emu = log_amp_emu[0]
    phase_emu = phase_emu[0]

    # Convert to complex strain
    amp_emu = 10.0 ** log_amp_emu
    h_emu = amp_emu * np.exp(-1j * phase_emu)

    # Generate LAL waveform
    # Convert eta to masses
    sqrt_term = np.sqrt(1 - 4 * eta)
    q = (1 - sqrt_term) / (1 + sqrt_term)
    m1 = M_total / (1 + q)
    m2 = M_total * q / (1 + q)

    # Convert Mf grid to physical frequencies
    Mf_grid = emulator.frequency_grid
    f_physical = np.array(geometric_to_physical_frequency(Mf_grid, M_total))

    # Generate LAL waveform with fine resolution
    lal_params = WaveformParameters(
        mass_1=m1,
        mass_2=m2,
        chi1z=chi1z,
        chi2z=chi2z,
        luminosity_distance=1.0,
        inclination=0.0,
        phase=0.0,
        f_min=float(f_physical[0] * 0.9),
        f_max=float(f_physical[-1] * 1.1),
        delta_f=0.1,
        approximant="IMRPhenomXPHM",
    )

    freqs_lal, hp_lal, _ = generate_fd_waveform(
        lal_params, mode_array=[(2, 2), (2, -2)]
    )

    # Convert LAL frequencies to geometric
    Mf_lal = np.array(physical_to_geometric_frequency(freqs_lal, M_total))

    # Interpolate LAL waveform to emulator grid
    valid_mask = np.abs(hp_lal) > 0
    h_lal_interp = np.interp(
        Mf_grid,
        Mf_lal[valid_mask],
        hp_lal[valid_mask],
        left=0.0,
        right=0.0,
    )

    # Align phases at peak amplitude
    peak_idx = np.argmax(np.abs(h_lal_interp))
    phase_lal = np.unwrap(np.angle(h_lal_interp))
    phase_lal = phase_lal - phase_lal[peak_idx]
    amp_lal = np.abs(h_lal_interp)
    h_lal_aligned = amp_lal * np.exp(-1j * phase_lal)

    # Also align emulator phase at peak
    h_emu_aligned = amp_emu * np.exp(-1j * (phase_emu - phase_emu[peak_idx]))

    # Load PSD (interpolate to physical frequencies)
    psd_freqs, psd_values = load_psd(psd_file)
    psd_interp = np.interp(f_physical, np.array(psd_freqs), np.array(psd_values))

    # Compute match
    match = float(compute_match(h_emu_aligned, h_lal_aligned, psd_interp, f_physical))
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
        "phase_emu": phase_emu - phase_emu[peak_idx],
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
    """
    Comprehensive validation of trained emulator.

    Parameters
    ----------
    emulator_path : str
        Path to trained emulator.
    data_path : str
        Path to training data HDF5.
    output_dir : str
        Output directory for plots.
    n_mismatch_samples : int
        Number of samples for mismatch calculation.
    psd_file : str
        Path to PSD file for mismatch.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Validating Waveform Emulator")
    print("=" * 60)

    # Load emulator
    print(f"\nLoading emulator from {emulator_path}...")
    emulator = GWEmulator.load(emulator_path)
    print(f"  Frequency grid: {len(emulator.frequency_grid)} points")
    print(f"  Amplitude PCA components: {emulator.n_amp_components}")
    print(f"  Phase PCA components: {emulator.n_phase_components}")

    # Load validation data
    print(f"\nLoading validation data from {data_path}...")
    with h5py.File(data_path, "r") as f:
        val_params = f["validation/parameters"][:]
        val_log_amp = f["validation/log_amplitude"][:]
        val_phase = f["validation/phase"][:]
        Mf_grid = f["frequency_grid"][:]

    print(f"  Validation samples: {len(val_params)}")

    # ==========================================================================
    # 1. Reconstruction Quality on Validation Set
    # ==========================================================================
    print("\n[1/3] Evaluating reconstruction quality...")

    # Get emulator predictions
    pred_log_amp, pred_phase = emulator.predict(val_params)

    # Amplitude errors
    amp_residuals = pred_log_amp - val_log_amp
    amp_mse = np.mean(amp_residuals ** 2)
    amp_max_error = np.max(np.abs(amp_residuals))
    amp_mean_abs_error = np.mean(np.abs(amp_residuals))

    print(f"  Amplitude (log10):")
    print(f"    MSE: {amp_mse:.2e}")
    print(f"    Max absolute error: {amp_max_error:.4f}")
    print(f"    Mean absolute error: {amp_mean_abs_error:.4f}")

    # Phase errors
    phase_residuals = pred_phase - val_phase
    phase_mse = np.mean(phase_residuals ** 2)
    phase_max_error = np.max(np.abs(phase_residuals))
    phase_mean_abs_error = np.mean(np.abs(phase_residuals))

    print(f"  Phase (rad):")
    print(f"    MSE: {phase_mse:.2e}")
    print(f"    Max absolute error: {phase_max_error:.4f} rad")
    print(f"    Mean absolute error: {phase_mean_abs_error:.4f} rad")

    # ==========================================================================
    # 2. Mismatch Against LAL
    # ==========================================================================
    print(f"\n[2/3] Computing mismatches against LAL ({n_mismatch_samples} samples)...")

    # Sample parameters for mismatch calculation
    rng = np.random.default_rng(42)
    mismatch_indices = rng.choice(len(val_params), size=min(n_mismatch_samples, len(val_params)), replace=False)

    mismatches = []
    mismatch_params = []
    failed = 0

    for idx in tqdm(mismatch_indices, desc="Mismatch calculation"):
        eta, chi1z, chi2z = val_params[idx]
        try:
            result = compute_mismatch_lal(
                emulator, eta, chi1z, chi2z, psd_file=psd_file
            )
            mismatches.append(result["mismatch"])
            mismatch_params.append([eta, chi1z, chi2z])
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"\n  Warning: Failed for eta={eta:.4f}, chi1z={chi1z:.4f}, chi2z={chi2z:.4f}: {e}")

    mismatches = np.array(mismatches)
    mismatch_params = np.array(mismatch_params)

    print(f"\n  Mismatch Statistics ({len(mismatches)} successful, {failed} failed):")
    print(f"    Median: {np.median(mismatches):.2e}")
    print(f"    Mean: {np.mean(mismatches):.2e}")
    print(f"    Max: {np.max(mismatches):.2e}")
    print(f"    Min: {np.min(mismatches):.2e}")
    print(f"    < 1e-3: {np.sum(mismatches < 1e-3) / len(mismatches) * 100:.1f}%")
    print(f"    < 1e-2: {np.sum(mismatches < 1e-2) / len(mismatches) * 100:.1f}%")

    # ==========================================================================
    # 3. Generate Plots
    # ==========================================================================
    print("\n[3/3] Generating validation plots...")

    # --- Plot 1: Reconstruction quality ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Amplitude residual distribution
    ax = axes[0, 0]
    ax.hist(amp_residuals.flatten(), bins=100, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('log₁₀(Amplitude) Residual')
    ax.set_ylabel('Density')
    ax.set_title(f'Amplitude Residuals (MSE={amp_mse:.2e})')
    ax.grid(True, alpha=0.3)

    # Phase residual distribution
    ax = axes[0, 1]
    ax.hist(phase_residuals.flatten(), bins=100, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Phase Residual (rad)')
    ax.set_ylabel('Density')
    ax.set_title(f'Phase Residuals (MSE={phase_mse:.2e})')
    ax.grid(True, alpha=0.3)

    # Per-frequency error
    ax = axes[0, 2]
    amp_mse_per_freq = np.mean(amp_residuals ** 2, axis=0)
    phase_mse_per_freq = np.mean(phase_residuals ** 2, axis=0)
    ax.semilogy(Mf_grid, amp_mse_per_freq, label='Amplitude', alpha=0.8)
    ax.semilogy(Mf_grid, phase_mse_per_freq, label='Phase', alpha=0.8)
    ax.set_xlabel('Geometric Frequency Mf')
    ax.set_ylabel('MSE')
    ax.set_xscale('log')
    ax.set_title('Per-Frequency MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sample waveform comparisons
    sample_indices = [0, len(val_params)//2, len(val_params)-1]
    for i, idx in enumerate(sample_indices):
        ax = axes[1, i]
        eta, chi1z, chi2z = val_params[idx]

        ax.plot(Mf_grid, val_log_amp[idx], 'b-', label='True', alpha=0.8, linewidth=2)
        ax.plot(Mf_grid, pred_log_amp[idx], 'r--', label='Emulator', alpha=0.8, linewidth=2)

        ax.set_xlabel('Mf')
        ax.set_ylabel('log₁₀(Amplitude)')
        ax.set_xscale('log')
        ax.set_title(f'η={eta:.3f}, χ₁={chi1z:.2f}, χ₂={chi2z:.2f}')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_quality.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/reconstruction_quality.png")

    # --- Plot 2: Mismatch distribution ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram
    ax = axes[0, 0]
    ax.hist(np.log10(mismatches), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.log10(1e-3), color='g', linestyle='--', label='Target (10⁻³)', linewidth=2)
    ax.axvline(np.log10(np.median(mismatches)), color='r', linestyle='-',
               label=f'Median ({np.median(mismatches):.1e})', linewidth=2)
    ax.set_xlabel('log₁₀(Mismatch)')
    ax.set_ylabel('Count')
    ax.set_title('Mismatch Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mismatch vs eta
    ax = axes[0, 1]
    sc = ax.scatter(mismatch_params[:, 0], mismatches, c=mismatch_params[:, 1],
                   cmap='coolwarm', alpha=0.7, s=20)
    ax.axhline(1e-3, color='g', linestyle='--', label='Target', linewidth=2)
    ax.set_xlabel('η (mass ratio)')
    ax.set_ylabel('Mismatch')
    ax.set_yscale('log')
    ax.set_title('Mismatch vs η (colored by χ₁z)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='χ₁z')

    # Mismatch vs chi1z
    ax = axes[1, 0]
    sc = ax.scatter(mismatch_params[:, 1], mismatches, c=mismatch_params[:, 0],
                   cmap='viridis', alpha=0.7, s=20)
    ax.axhline(1e-3, color='g', linestyle='--', label='Target', linewidth=2)
    ax.set_xlabel('χ₁z (primary spin)')
    ax.set_ylabel('Mismatch')
    ax.set_yscale('log')
    ax.set_title('Mismatch vs χ₁z (colored by η)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='η')

    # 2D parameter space
    ax = axes[1, 1]
    sc = ax.scatter(mismatch_params[:, 1], mismatch_params[:, 2],
                   c=np.log10(mismatches), cmap='RdYlGn_r', alpha=0.7, s=20)
    ax.set_xlabel('χ₁z')
    ax.set_ylabel('χ₂z')
    ax.set_title('Mismatch in Spin Space')
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='log₁₀(Mismatch)')

    plt.tight_layout()
    plt.savefig(output_dir / "mismatch_analysis.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/mismatch_analysis.png")

    # --- Plot 3: Example waveform comparisons ---
    # Find worst and best cases
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

        # Get full comparison
        result = compute_mismatch_lal(emulator, eta, chi1z, chi2z, psd_file=psd_file)

        # Amplitude
        ax = axes[row, 0]
        ax.plot(Mf_grid, np.log10(result["amp_lal"] + 1e-100), 'b-', label='LAL', linewidth=2)
        ax.plot(Mf_grid, np.log10(result["amp_emu"] + 1e-100), 'r--', label='Emulator', linewidth=2)
        ax.set_xlabel('Mf')
        ax.set_ylabel('log₁₀(Amplitude)')
        ax.set_xscale('log')
        ax.set_title(f'{label} Case: η={eta:.3f}, χ₁={chi1z:.2f}, χ₂={chi2z:.2f}\nMismatch={mm:.2e}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Phase
        ax = axes[row, 1]
        ax.plot(Mf_grid, result["phase_lal"], 'b-', label='LAL', linewidth=2)
        ax.plot(Mf_grid, result["phase_emu"], 'r--', label='Emulator', linewidth=2)
        ax.set_xlabel('Mf')
        ax.set_ylabel('Phase (rad)')
        ax.set_xscale('log')
        ax.set_title(f'Phase Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "waveform_comparisons.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/waveform_comparisons.png")

    # --- Plot 4: Summary statistics ---
    fig, ax = plt.subplots(figsize=(8, 6))

    stats_text = f"""
Emulator Validation Summary
===========================

Reconstruction Quality (Validation Set)
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
  Amplitude PCA: {emulator.n_amp_components} components
  Phase PCA: {emulator.n_phase_components} components
  Frequency grid: {len(emulator.frequency_grid)} points
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "validation_summary.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/validation_summary.png")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)

    return {
        "amp_mse": amp_mse,
        "phase_mse": phase_mse,
        "median_mismatch": np.median(mismatches),
        "max_mismatch": np.max(mismatches),
        "mismatches": mismatches,
        "mismatch_params": mismatch_params,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate waveform emulator")
    parser.add_argument("--emulator", type=str, default="outputs/emulator_22mode.pkl",
                        help="Path to trained emulator")
    parser.add_argument("--data", type=str, default="data/waveforms_22mode.h5",
                        help="Path to training data")
    parser.add_argument("--output", type=str, default="outputs/validation",
                        help="Output directory for plots")
    parser.add_argument("--n_mismatch", type=int, default=200,
                        help="Number of mismatch samples")
    parser.add_argument("--psd", type=str, default="psds/ET-D-psd.txt",
                        help="Path to PSD file")
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
