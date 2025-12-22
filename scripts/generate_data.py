#!/usr/bin/env python
"""
Generate training data for gravitational waveform emulator.

This script generates waveforms using LALSimulation and stores them in HDF5 format
for training a neural network emulator. We use direct LAL output with no interpolation.

Starter case: Aligned spins (no precession), 22 mode only.

Usage:
    python scripts/generate_data.py                    # Default: 10k train, 1k val
    python scripts/generate_data.py --n_train 100000   # Full dataset
    python scripts/generate_data.py --test             # Quick test with 100 samples
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple
import time

import h5py
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.waveforms import WaveformParameters, generate_fd_waveform


# =============================================================================
# Configuration
# =============================================================================

# Parameter bounds for aligned-spin BBH
PARAM_BOUNDS = {
    "eta": (0.05, 0.25),      # Symmetric mass ratio (q=18 to q=1)
    "chi1z": (-0.99, 0.99),   # Primary aligned spin
    "chi2z": (-0.99, 0.99),   # Secondary aligned spin
}

# Fixed frequency grid (physical Hz)
# Using a reference mass of 50 Msun, these correspond roughly to Mf in [0.003, 0.25]
F_MIN = 10.0     # Hz - start of waveform
F_MAX = 1024.0   # Hz - past ringdown for most systems
DELTA_F = 1.0    # Hz - frequency spacing

# Reference total mass for waveform generation
M_REF = 50.0  # Solar masses

# Mode selection for 22-only
MODE_22_ONLY = [(2, 2), (2, -2)]


# =============================================================================
# Parameter Sampling
# =============================================================================

def sample_parameters_lhs(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Sample parameter space using Latin Hypercube Sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    params : np.ndarray
        Shape (n_samples, 3) with columns [eta, chi1z, chi2z].
    """
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    samples = sampler.random(n_samples)

    # Scale to parameter bounds
    bounds = np.array([
        PARAM_BOUNDS["eta"],
        PARAM_BOUNDS["chi1z"],
        PARAM_BOUNDS["chi2z"],
    ])

    params = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
    return params.astype(np.float64)


def eta_to_masses(eta: float, M_total: float) -> Tuple[float, float]:
    """
    Convert symmetric mass ratio to component masses.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio eta = m1*m2/(m1+m2)^2, in [0, 0.25].
    M_total : float
        Total mass m1 + m2.

    Returns
    -------
    m1, m2 : float
        Component masses with m1 >= m2.
    """
    sqrt_term = np.sqrt(1 - 4 * eta)
    q = (1 - sqrt_term) / (1 + sqrt_term)

    m1 = M_total / (1 + q)
    m2 = M_total * q / (1 + q)

    return m1, m2


# =============================================================================
# Waveform Generation
# =============================================================================

def generate_waveform_direct(
    eta: float,
    chi1z: float,
    chi2z: float,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
    delta_f: float = DELTA_F,
    M_total: float = M_REF,
    mode_array: list = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a waveform directly from LAL with no interpolation.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    chi1z, chi2z : float
        Aligned spin components.
    f_min, f_max, delta_f : float
        Frequency grid parameters in Hz.
    M_total : float
        Reference total mass in solar masses.
    mode_array : list, optional
        Modes to include. Default is 22-only.

    Returns
    -------
    frequencies : np.ndarray
        Frequency grid in Hz (from LAL).
    log_amplitude : np.ndarray
        Log10 of strain amplitude.
    phase : np.ndarray
        Unwrapped phase, aligned at peak amplitude.
    """
    if mode_array is None:
        mode_array = MODE_22_ONLY

    # Convert eta to component masses
    m1, m2 = eta_to_masses(eta, M_total)

    # Create waveform parameters
    params = WaveformParameters(
        mass_1=m1,
        mass_2=m2,
        chi1z=chi1z,
        chi2z=chi2z,
        luminosity_distance=1.0,  # Normalization handled separately
        inclination=0.0,          # Face-on for simplicity (hp only)
        phase=0.0,
        f_min=f_min,
        f_max=f_max,
        delta_f=delta_f,
        approximant="IMRPhenomXPHM",
    )

    # Generate waveform directly from LAL (disable multibanding for smooth waveforms)
    freqs, hp, hc = generate_fd_waveform(
        params, mode_array=mode_array, disable_multibanding=True
    )

    # Extract amplitude and phase directly from LAL output
    amplitude = np.abs(hp)
    phase = np.unwrap(np.angle(hp))

    # Handle zeros: set minimum amplitude floor for log
    amp_floor = 1e-100
    log_amplitude = np.log10(np.maximum(amplitude, amp_floor))

    # Align phase at peak amplitude
    peak_idx = np.argmax(amplitude)
    phase = phase - phase[peak_idx]

    return freqs, log_amplitude, phase


# =============================================================================
# Main Data Generation
# =============================================================================

def generate_dataset(
    n_samples: int,
    output_path: str,
    seed: int = 42,
    dataset_name: str = "train",
) -> None:
    """
    Generate a full dataset of waveforms.

    Parameters
    ----------
    n_samples : int
        Number of waveforms to generate.
    output_path : str
        Path to output HDF5 file.
    seed : int
        Random seed.
    dataset_name : str
        Name for the dataset group (e.g., "train", "validation").
    """
    print(f"Generating {n_samples} waveforms for {dataset_name} set...")

    # Sample parameters
    params = sample_parameters_lhs(n_samples, seed=seed)

    # Generate first waveform to get frequency grid size
    eta0, chi1z0, chi2z0 = params[0]
    freqs, _, _ = generate_waveform_direct(eta0, chi1z0, chi2z0)
    n_freq = len(freqs)
    print(f"Frequency grid: {n_freq} points from {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")

    # Allocate output arrays
    amplitudes = np.zeros((n_samples, n_freq), dtype=np.float64)
    phases = np.zeros((n_samples, n_freq), dtype=np.float64)
    success_mask = np.zeros(n_samples, dtype=bool)

    # Generate waveforms with progress bar
    start_time = time.time()
    for i in tqdm(range(n_samples), desc=f"Generating {dataset_name}"):
        eta, chi1z, chi2z = params[i]
        try:
            f, log_amp, phase = generate_waveform_direct(eta, chi1z, chi2z)
            amplitudes[i] = log_amp
            phases[i] = phase
            success_mask[i] = True
        except Exception as e:
            print(f"\nWarning: Failed for sample {i}: {e}")
            amplitudes[i] = np.nan
            phases[i] = np.nan
            success_mask[i] = False

    elapsed = time.time() - start_time
    n_success = np.sum(success_mask)
    print(f"Generated {n_success}/{n_samples} waveforms in {elapsed:.1f}s ({elapsed/n_samples:.2f}s per waveform)")

    # Filter to successful samples only
    params = params[success_mask]
    amplitudes = amplitudes[success_mask]
    phases = phases[success_mask]

    # Save to HDF5
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mode = "a" if os.path.exists(output_path) else "w"

    with h5py.File(output_path, mode) as f:
        # Create or overwrite group
        if dataset_name in f:
            del f[dataset_name]
        grp = f.create_group(dataset_name)

        # Store data
        grp.create_dataset("parameters", data=params, compression="gzip")
        grp.create_dataset("log_amplitude", data=amplitudes, compression="gzip")
        grp.create_dataset("phase", data=phases, compression="gzip")

        # Store metadata in root (only once)
        if "frequency_grid" not in f:
            f.create_dataset("frequency_grid", data=freqs)
            f.attrs["param_names"] = ["eta", "chi1z", "chi2z"]
            f.attrs["param_bounds_eta"] = PARAM_BOUNDS["eta"]
            f.attrs["param_bounds_chi1z"] = PARAM_BOUNDS["chi1z"]
            f.attrs["param_bounds_chi2z"] = PARAM_BOUNDS["chi2z"]
            f.attrs["f_min"] = F_MIN
            f.attrs["f_max"] = F_MAX
            f.attrs["delta_f"] = DELTA_F
            f.attrs["M_ref"] = M_REF
            f.attrs["n_freq"] = n_freq
            f.attrs["approximant"] = "IMRPhenomXPHM"
            f.attrs["modes"] = "22_only"
            f.attrs["description"] = "Aligned spin BBH waveforms, 22 mode only, direct LAL output"

    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate waveform training data")
    parser.add_argument("--n_train", type=int, default=10000,
                        help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=1000,
                        help="Number of validation samples")
    parser.add_argument("--output", type=str, default="data/waveforms_22mode.h5",
                        help="Output HDF5 file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode with 100 train, 10 val")
    args = parser.parse_args()

    if args.test:
        args.n_train = 100
        args.n_val = 10
        args.output = "data/waveforms_22mode_test.h5"

    # Generate training set
    generate_dataset(
        n_samples=args.n_train,
        output_path=args.output,
        seed=args.seed,
        dataset_name="train",
    )

    # Generate validation set (different seed)
    generate_dataset(
        n_samples=args.n_val,
        output_path=args.output,
        seed=args.seed + 1000,
        dataset_name="validation",
    )

    # Print summary
    with h5py.File(args.output, "r") as f:
        print("\n=== Dataset Summary ===")
        print(f"File: {args.output}")
        freqs = f['frequency_grid'][:]
        print(f"Frequency grid: {len(freqs)} points from {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")
        print(f"Training samples: {f['train/parameters'].shape[0]}")
        print(f"Validation samples: {f['validation/parameters'].shape[0]}")
        print(f"Parameter bounds:")
        for name in ["eta", "chi1z", "chi2z"]:
            bounds = f.attrs[f"param_bounds_{name}"]
            print(f"  {name}: [{bounds[0]:.2f}, {bounds[1]:.2f}]")


if __name__ == "__main__":
    main()
