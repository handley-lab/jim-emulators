#!/usr/bin/env python
"""
Validate generated waveform data.

This script performs comprehensive validation of the generated training data:
1. Checks for NaN values and their locations
2. Validates amplitude and phase statistics
3. Verifies phase alignment at reference frequency
4. Checks parameter space coverage
5. Tests data continuity
6. Compares stored data against fresh LAL generation

Usage:
    python scripts/validate_data.py data/waveforms_22mode.h5
    python scripts/validate_data.py data/waveforms_22mode_test.h5
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.waveforms import WaveformParameters, generate_fd_mode

MTSUN_SI = 4.925491025543576e-6
M_REF = 50.0


def validate_dataset(filepath: str, make_plots: bool = True) -> bool:
    """
    Validate a waveform dataset.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.
    make_plots : bool
        Whether to generate validation plots.

    Returns
    -------
    valid : bool
        True if all validation checks pass.
    """
    print("=" * 60)
    print("VALIDATION OF GENERATED WAVEFORM DATA")
    print("=" * 60)
    print(f"File: {filepath}")

    all_checks_passed = True

    with h5py.File(filepath, 'r') as f:
        # Load data
        Mf_grid = f['frequency_grid'][:]
        train_params = f['train/parameters'][:]
        train_log_amp = f['train/log_amplitude'][:]
        train_phase = f['train/phase'][:]
        Mf_ref = f.attrs['Mf_ref']

        # 1. Dataset overview
        print("\n1. DATASET OVERVIEW")
        print("-" * 40)
        print(f"Training samples: {len(train_params)}")
        if 'validation' in f:
            print(f"Validation samples: {len(f['validation/parameters'])}")
        print(f"Frequency grid: {len(Mf_grid)} points")
        print(f"Mf range: [{Mf_grid[0]:.6f}, {Mf_grid[-1]:.4f}]")
        print(f"Reference Mf: {Mf_ref}")

        # 2. NaN analysis
        print("\n2. NAN ANALYSIS")
        print("-" * 40)
        nan_per_sample = np.sum(np.isnan(train_log_amp), axis=1)
        samples_with_nan = np.sum(nan_per_sample > 0)
        print(f"Samples with any NaN: {samples_with_nan}/{len(train_params)}")
        print(f"Average NaN per sample: {np.mean(nan_per_sample):.1f}")
        print(f"Max NaN per sample: {np.max(nan_per_sample)}")

        # NaN location
        sample_nan_mask = np.isnan(train_log_amp[0])
        nan_at_start = np.sum(sample_nan_mask[:100])
        nan_at_end = np.sum(sample_nan_mask[-100:])
        print(f"NaNs in first 100 bins: {nan_at_start}, last 100 bins: {nan_at_end}")

        # 3. Amplitude statistics
        print("\n3. AMPLITUDE STATISTICS")
        print("-" * 40)
        valid_amp = train_log_amp[~np.isnan(train_log_amp)]
        print(f"Log10 amplitude range: [{valid_amp.min():.2f}, {valid_amp.max():.2f}]")
        print(f"Log10 amplitude mean: {valid_amp.mean():.2f}, std: {valid_amp.std():.2f}")
        print(f"Physical amplitude range: [{10**valid_amp.min():.2e}, {10**valid_amp.max():.2e}]")

        # 4. Phase statistics
        print("\n4. PHASE STATISTICS")
        print("-" * 40)
        valid_phase = train_phase[~np.isnan(train_phase)]
        print(f"Phase range: [{valid_phase.min():.1f}, {valid_phase.max():.1f}] rad")
        print(f"Phase mean: {valid_phase.mean():.1f}, std: {valid_phase.std():.1f}")

        # Phase at reference frequency
        ref_idx = np.argmin(np.abs(Mf_grid - Mf_ref))
        phase_at_ref = train_phase[:, ref_idx]
        ref_mean = np.nanmean(phase_at_ref)
        ref_std = np.nanstd(phase_at_ref)
        ref_max_dev = np.nanmax(np.abs(phase_at_ref))
        print(f"\nPhase at Mf_ref={Mf_ref}:")
        print(f"  mean: {ref_mean:.6f} (should be ~0)")
        print(f"  std: {ref_std:.6f}")
        print(f"  max deviation: {ref_max_dev:.6f}")

        if ref_max_dev > 1e-6:
            print("  ⚠ Phase alignment issue!")
            all_checks_passed = False
        else:
            print("  ✓ Phase correctly aligned at reference")

        # 5. Parameter space coverage
        print("\n5. PARAMETER SPACE COVERAGE")
        print("-" * 40)
        eta, chi1z, chi2z = train_params.T
        print(f"eta: [{eta.min():.3f}, {eta.max():.3f}], mean={eta.mean():.3f}")
        print(f"chi1z: [{chi1z.min():.3f}, {chi1z.max():.3f}], mean={chi1z.mean():.3f}")
        print(f"chi2z: [{chi2z.min():.3f}, {chi2z.max():.3f}], mean={chi2z.mean():.3f}")

        print("\nQuartile coverage:")
        for i, name in enumerate(['eta', 'chi1z', 'chi2z']):
            vals = train_params[:, i]
            quartiles = np.percentile(vals, [25, 50, 75])
            print(f"  {name}: Q1={quartiles[0]:.3f}, Q2={quartiles[1]:.3f}, Q3={quartiles[2]:.3f}")

        # 6. Data continuity
        print("\n6. DATA CONTINUITY")
        print("-" * 40)
        amp_diff = np.diff(train_log_amp, axis=1)
        max_amp_jump = np.nanmax(np.abs(amp_diff))
        print(f"Max log-amplitude jump between adjacent bins: {max_amp_jump:.3f}")

        phase_diff = np.diff(train_phase, axis=1)
        max_phase_jump = np.nanmax(np.abs(phase_diff))
        print(f"Max phase jump between adjacent bins: {max_phase_jump:.3f} rad")

        if max_phase_jump > np.pi:
            print(f"  Note: Phase jump > pi may indicate unwrapping issues at high Mf")

        # 7. Verification against LAL
        print("\n7. VERIFICATION: Compare against fresh LAL generation")
        print("-" * 40)
        test_indices = [0, len(train_params)//2, len(train_params)-1]

        for idx in test_indices:
            params = train_params[idx]
            eta, chi1z, chi2z = params
            print(f"\nSample {idx}: eta={eta:.4f}, chi1z={chi1z:.4f}, chi2z={chi2z:.4f}")

            # Regenerate from parameters
            sqrt_term = np.sqrt(1 - 4 * eta)
            q = (1 - sqrt_term) / (1 + sqrt_term)
            m1 = M_REF / (1 + q)
            m2 = M_REF * q / (1 + q)

            # Convert Mf to physical frequency for LAL
            f_physical = Mf_grid / (M_REF * MTSUN_SI)
            f_min = f_physical[0]
            f_max = f_physical[-1]
            f_ref = Mf_ref / (M_REF * MTSUN_SI)
            delta_f = min(0.1, (f_max - f_min) / 10000)

            wf_params = WaveformParameters(
                mass_1=m1, mass_2=m2,
                chi1z=chi1z, chi2z=chi2z,
                luminosity_distance=1.0,
                f_min=max(f_min * 0.9, 1.0),
                f_max=f_max * 1.1,
                delta_f=delta_f,
                f_ref=f_ref,
            )

            freqs, h22 = generate_fd_mode(wf_params, ell=2, emm=2)

            # Convert to geometric and interpolate
            Mf_gen = freqs * M_REF * MTSUN_SI
            amp_gen = np.abs(h22)
            phase_gen = np.unwrap(np.angle(h22))

            valid_mask = amp_gen > 0

            # Interpolate to our grid
            log_amp_fresh = np.interp(
                Mf_grid, Mf_gen[valid_mask], np.log10(amp_gen[valid_mask] + 1e-100),
                left=np.nan, right=np.nan
            )
            phase_fresh = np.interp(
                Mf_grid, Mf_gen[valid_mask], phase_gen[valid_mask],
                left=np.nan, right=np.nan
            )

            # Align phase at reference
            if not np.isnan(phase_fresh[ref_idx]):
                phase_fresh = phase_fresh - phase_fresh[ref_idx]

            # Compare
            valid = ~np.isnan(train_log_amp[idx]) & ~np.isnan(log_amp_fresh)

            amp_diff = np.abs(train_log_amp[idx][valid] - log_amp_fresh[valid])
            phase_diff = np.abs(train_phase[idx][valid] - phase_fresh[valid])

            print(f"  Amplitude: max diff={amp_diff.max():.6f}, mean={amp_diff.mean():.6f}")
            print(f"  Phase: max diff={phase_diff.max():.6f} rad, mean={phase_diff.mean():.6f}")

            if amp_diff.max() < 1e-6 and phase_diff.max() < 1e-6:
                print("  ✓ Perfect match!")
            elif amp_diff.max() < 1e-3 and phase_diff.max() < 1e-3:
                print("  ✓ Good match (< 1e-3)")
            else:
                print("  ⚠ Mismatch detected!")
                all_checks_passed = False

    # Generate plots if requested
    if make_plots:
        print("\n8. GENERATING VALIDATION PLOTS")
        print("-" * 40)
        generate_validation_plots(filepath)

    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
    else:
        print("⚠ SOME VALIDATION CHECKS FAILED")
    print("=" * 60)

    return all_checks_passed


def generate_validation_plots(filepath: str):
    """Generate validation plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    with h5py.File(filepath, 'r') as f:
        Mf_grid = f['frequency_grid'][:]
        train_params = f['train/parameters'][:]
        train_log_amp = f['train/log_amplitude'][:]
        train_phase = f['train/phase'][:]
        Mf_ref = f.attrs['Mf_ref']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot amplitude samples
    ax = axes[0, 0]
    n_samples = min(20, len(train_params))
    for i in np.linspace(0, len(train_params)-1, n_samples, dtype=int):
        ax.plot(Mf_grid, train_log_amp[i], alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Mf')
    ax.set_ylabel('log10(Amplitude)')
    ax.set_xscale('log')
    ax.set_title(f'Log Amplitude ({n_samples} samples)')
    ax.axvline(x=Mf_ref, color='r', linestyle='--', alpha=0.5, label='Mf_ref')

    # Plot phase samples
    ax = axes[0, 1]
    for i in np.linspace(0, len(train_params)-1, n_samples, dtype=int):
        ax.plot(Mf_grid, train_phase[i], alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Mf')
    ax.set_ylabel('Phase (rad)')
    ax.set_xscale('log')
    ax.set_title(f'Unwrapped Phase ({n_samples} samples)')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=Mf_ref, color='r', linestyle='--', alpha=0.5)

    # Parameter space coverage
    ax = axes[0, 2]
    scatter = ax.scatter(train_params[:, 0], train_params[:, 1], c=train_params[:, 2],
                         cmap='coolwarm', s=3, alpha=0.5)
    ax.set_xlabel('eta')
    ax.set_ylabel('chi1z')
    ax.set_title('Parameter space (color=chi2z)')
    plt.colorbar(scatter, ax=ax, label='chi2z')

    # Amplitude histogram
    ax = axes[1, 0]
    valid_amp = train_log_amp[~np.isnan(train_log_amp)]
    ax.hist(valid_amp, bins=100, alpha=0.7)
    ax.set_xlabel('log10(Amplitude)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of log amplitudes')

    # Phase histogram
    ax = axes[1, 1]
    valid_phase = train_phase[~np.isnan(train_phase)]
    ax.hist(valid_phase, bins=100, alpha=0.7)
    ax.set_xlabel('Phase (rad)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of phases')

    # Phase at reference frequency
    ax = axes[1, 2]
    ref_idx = np.argmin(np.abs(Mf_grid - Mf_ref))
    phase_at_ref = train_phase[:, ref_idx]
    ax.hist(phase_at_ref[~np.isnan(phase_at_ref)], bins=50, alpha=0.7)
    ax.set_xlabel('Phase at Mf_ref (rad)')
    ax.set_ylabel('Count')
    ax.set_title(f'Phase at Mf_ref={Mf_ref}')
    ax.axvline(x=0, color='r', linestyle='--', label='Expected (0)')
    ax.legend()

    plt.tight_layout()
    output_path = filepath.replace('.h5', '_validation.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved validation plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate waveform training data")
    parser.add_argument("filepath", type=str, help="Path to HDF5 file")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    valid = validate_dataset(args.filepath, make_plots=not args.no_plots)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
