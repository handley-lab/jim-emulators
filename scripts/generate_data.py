#!/usr/bin/env python
"""
Generate training data for gravitational waveform emulator.

This script generates waveforms using LALSimulation and stores them in HDF5 format
for training a neural network emulator. The waveforms are parameterized in geometric
units (Mf) for mass-independence.

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
from jim_emulators.waveforms.utils import (
    get_geometric_frequency_grid,
    physical_to_geometric_frequency,
    geometric_to_physical_frequency,
)


# =============================================================================
# Configuration
# =============================================================================

# Parameter bounds for aligned-spin BBH
PARAM_BOUNDS = {
    "eta": (0.05, 0.25),      # Symmetric mass ratio (q=18 to q=1)
    "chi1z": (-0.99, 0.99),   # Primary aligned spin
    "chi2z": (-0.99, 0.99),   # Secondary aligned spin
}

# Geometric frequency grid
MF_MIN = 0.003   # Start early enough to capture inspiral
MF_MAX = 0.25    # Past ringdown for most systems
N_FREQ = 1000    # Number of frequency points

# Reference total mass for waveform generation (cancels out in geometric units)
# Using 50 Msun gives reasonable frequency range
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
    # From eta, compute mass ratio q = m2/m1 <= 1
    # eta = q / (1+q)^2
    # Solving: q = (1 - 2*eta - sqrt(1 - 4*eta)) / (2*eta)
    # But simpler: q = (1 - sqrt(1 - 4*eta)) / (1 + sqrt(1 - 4*eta))
    sqrt_term = np.sqrt(1 - 4 * eta)
    q = (1 - sqrt_term) / (1 + sqrt_term)

    m1 = M_total / (1 + q)
    m2 = M_total * q / (1 + q)

    return m1, m2


# =============================================================================
# Waveform Generation
# =============================================================================

def generate_waveform_geometric(
    eta: float,
    chi1z: float,
    chi2z: float,
    Mf_grid: np.ndarray,
    M_total: float = M_REF,
    mode_array: list = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a waveform on a geometric frequency grid.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    chi1z, chi2z : float
        Aligned spin components.
    Mf_grid : np.ndarray
        Geometric frequency grid.
    M_total : float
        Reference total mass (for waveform generation).
    mode_array : list, optional
        Modes to include. Default is 22-only.

    Returns
    -------
    amplitude : np.ndarray
        Log10 amplitude on Mf grid.
    phase : np.ndarray
        Unwrapped, aligned phase on Mf grid.
    """
    if mode_array is None:
        mode_array = MODE_22_ONLY

    # Convert eta to component masses
    m1, m2 = eta_to_masses(eta, M_total)

    # Convert geometric to physical frequencies
    f_physical = np.array(geometric_to_physical_frequency(Mf_grid, M_total))
    f_min = float(f_physical[0])
    f_max = float(f_physical[-1])

    # Use fine delta_f for accuracy, we'll interpolate
    delta_f = 0.1  # Hz

    # Create waveform parameters
    params = WaveformParameters(
        mass_1=m1,
        mass_2=m2,
        chi1z=chi1z,
        chi2z=chi2z,
        luminosity_distance=1.0,  # Amplitude in geometric units anyway
        inclination=0.0,          # Face-on for simplicity (hp only)
        phase=0.0,
        f_min=f_min * 0.9,        # Start a bit earlier for interpolation
        f_max=f_max * 1.1,        # End a bit later
        delta_f=delta_f,
        approximant="IMRPhenomXPHM",
    )

    # Generate waveform
    freqs, hp, hc = generate_fd_waveform(params, mode_array=mode_array)

    # Convert generated frequencies to geometric
    Mf_generated = np.array(physical_to_geometric_frequency(freqs, M_total))

    # Interpolate to target grid
    # We need both amplitude and phase
    amplitude = np.abs(hp)
    phase = np.unwrap(np.angle(hp))

    # Find valid range (non-zero amplitude)
    valid_mask = amplitude > 0
    if not np.any(valid_mask):
        raise ValueError(f"No valid waveform data for eta={eta}, chi1z={chi1z}, chi2z={chi2z}")

    # Interpolate amplitude (in log space for stability)
    log_amp_interp = np.interp(
        Mf_grid,
        Mf_generated[valid_mask],
        np.log10(amplitude[valid_mask] + 1e-100),  # Avoid log(0)
        left=np.nan,
        right=np.nan,
    )

    # Interpolate phase
    phase_interp = np.interp(
        Mf_grid,
        Mf_generated[valid_mask],
        phase[valid_mask],
        left=np.nan,
        right=np.nan,
    )

    # Align phase: subtract linear trend and set phase=0 at peak amplitude
    # Find peak in the valid region
    valid_interp = ~np.isnan(log_amp_interp)
    if np.any(valid_interp):
        peak_idx = np.argmax(np.where(valid_interp, log_amp_interp, -np.inf))
        phase_interp = phase_interp - phase_interp[peak_idx]

    return log_amp_interp, phase_interp


def process_single_waveform(args):
    """Worker function for parallel processing."""
    idx, eta, chi1z, chi2z, Mf_grid = args
    try:
        log_amp, phase = generate_waveform_geometric(eta, chi1z, chi2z, Mf_grid)
        return idx, log_amp, phase, True
    except Exception as e:
        print(f"Warning: Failed for idx={idx}, eta={eta:.4f}, chi1z={chi1z:.4f}, chi2z={chi2z:.4f}: {e}")
        return idx, None, None, False


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

    # Create geometric frequency grid
    Mf_grid = np.array(get_geometric_frequency_grid(N_FREQ, MF_MIN, MF_MAX))

    # Allocate output arrays
    amplitudes = np.zeros((n_samples, N_FREQ), dtype=np.float64)
    phases = np.zeros((n_samples, N_FREQ), dtype=np.float64)
    success_mask = np.zeros(n_samples, dtype=bool)

    # Generate waveforms with progress bar
    start_time = time.time()
    for i in tqdm(range(n_samples), desc=f"Generating {dataset_name}"):
        eta, chi1z, chi2z = params[i]
        try:
            log_amp, phase = generate_waveform_geometric(eta, chi1z, chi2z, Mf_grid)
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
            f.create_dataset("frequency_grid", data=Mf_grid)
            f.attrs["param_names"] = ["eta", "chi1z", "chi2z"]
            f.attrs["param_bounds_eta"] = PARAM_BOUNDS["eta"]
            f.attrs["param_bounds_chi1z"] = PARAM_BOUNDS["chi1z"]
            f.attrs["param_bounds_chi2z"] = PARAM_BOUNDS["chi2z"]
            f.attrs["Mf_min"] = MF_MIN
            f.attrs["Mf_max"] = MF_MAX
            f.attrs["n_freq"] = N_FREQ
            f.attrs["approximant"] = "IMRPhenomXPHM"
            f.attrs["modes"] = "22_only"
            f.attrs["description"] = "Aligned spin BBH waveforms, 22 mode only"

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
        print(f"Frequency grid: {f['frequency_grid'].shape[0]} points in Mf=[{MF_MIN}, {MF_MAX}]")
        print(f"Training samples: {f['train/parameters'].shape[0]}")
        print(f"Validation samples: {f['validation/parameters'].shape[0]}")
        print(f"Parameter bounds:")
        for name in ["eta", "chi1z", "chi2z"]:
            bounds = f.attrs[f"param_bounds_{name}"]
            print(f"  {name}: [{bounds[0]:.2f}, {bounds[1]:.2f}]")


if __name__ == "__main__":
    main()
