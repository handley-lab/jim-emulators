#!/usr/bin/env python
"""
Generate training data for gravitational waveform emulator.

This script generates individual spherical harmonic modes h_lm using LALSimulation
and stores them in HDF5 format for training a neural network emulator. The waveforms
are parameterized in geometric units (Mf) for mass-independence.

Following the framework in theory/frequency-domain-emulation.tex:
- Extracts individual modes h_lm(f), not combined polarizations
- Uses geometric frequency Mf = M*f as the domain variable
- Stores log-amplitude and unwrapped phase with proper conventions
- Phase aligned at reference frequency: Phi(Mf_ref) = 0

Frequency Grid Notes:
    The theory document recommends log-spaced grids in Mf for training, which
    better captures the inspiral dynamics. However, likelihood inference codes
    (e.g., jim) may require linear grids in physical frequency f. This script
    generates data on a log-spaced Mf grid for training. At inference time,
    the emulator can be evaluated at arbitrary Mf values (converted from
    physical f given the source mass M).

Usage:
    python scripts/generate_data.py                    # Default: 10k train, 1k val
    python scripts/generate_data.py --n_train 100000   # Large dataset
    python scripts/generate_data.py --test             # Quick test with 100 samples
    python scripts/generate_data.py --mode 3 3         # Generate (3,3) mode
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, List
import time

import h5py
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.waveforms import generate_fd_mode_at_frequencies
from jim_emulators.waveforms.utils import (
    get_geometric_frequency_grid,
    geometric_to_physical_frequency,
)

# Physical constants
MTSUN_SI = 4.925491025543576e-6  # G*Msun/c^3 in seconds


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
# Note: At very low Mf, the phase evolves rapidly and can alias on a log-spaced grid.
# MF_MIN = 0.004 avoids aliasing while covering the detector-sensitive band:
#   - For M = 50 Msun: f_min ~ 16 Hz
#   - For M = 20 Msun: f_min ~ 40 Hz
MF_MIN = 0.004    # Avoids phase aliasing at low frequencies
MF_MAX = 0.3      # Well past ringdown
N_FREQ = 2000     # Number of frequency points (log-spaced)

# Reference frequency for phase alignment (in geometric units)
# Must be safely above MF_MIN to ensure well-resolved phase
MF_REF = 0.006    # Reference point where Phi = 0 (f ~ 24 Hz for 50 Msun)

# Reference total mass for waveform generation
# The specific value doesn't matter for the dimensionless shape function,
# but affects the physical frequency range we generate at
M_REF = 50.0  # Solar masses


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
    # q = (1 - sqrt(1 - 4*eta)) / (1 + sqrt(1 - 4*eta))
    sqrt_term = np.sqrt(1 - 4 * eta)
    q = (1 - sqrt_term) / (1 + sqrt_term)

    m1 = M_total / (1 + q)
    m2 = M_total * q / (1 + q)

    return m1, m2


# =============================================================================
# Waveform Generation
# =============================================================================

def generate_mode_on_geometric_grid(
    eta: float,
    chi1z: float,
    chi2z: float,
    Mf_grid: np.ndarray,
    Mf_ref: float,
    ell: int = 2,
    emm: int = 2,
    M_total: float = M_REF,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single mode h_lm on a geometric frequency grid.

    Evaluates the waveform directly at the specified Mf grid points
    (no interpolation). Uses LAL's frequency sequence function for
    exact evaluation at arbitrary frequencies.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    chi1z, chi2z : float
        Aligned spin components.
    Mf_grid : np.ndarray
        Target geometric frequency grid (can be log-spaced or arbitrary).
    Mf_ref : float
        Reference geometric frequency for phase alignment.
    ell, emm : int
        Spherical harmonic mode numbers.
    M_total : float
        Reference total mass for converting Mf to physical f.

    Returns
    -------
    log_amplitude : np.ndarray
        Log10 of amplitude |h_lm| on Mf grid.
    phase : np.ndarray
        Unwrapped phase of h_lm on Mf grid, aligned so phase(Mf_ref) = 0.
    """
    # Convert eta to component masses
    m1, m2 = eta_to_masses(eta, M_total)

    # Convert geometric to physical frequencies
    f_physical = np.array(geometric_to_physical_frequency(Mf_grid, M_total))

    # Reference frequency in physical units
    f_ref_physical = float(geometric_to_physical_frequency(np.array([Mf_ref]), M_total)[0])

    # Generate mode at exact frequencies (no interpolation)
    h_lm = generate_fd_mode_at_frequencies(
        frequencies=f_physical,
        mass_1=m1,
        mass_2=m2,
        chi1z=chi1z,
        chi2z=chi2z,
        ell=ell,
        emm=emm,
        luminosity_distance=1.0,  # Reference distance
        phase=0.0,
        f_ref=f_ref_physical,
    )

    # Extract amplitude and phase
    amplitude = np.abs(h_lm)
    phase = np.unwrap(np.angle(h_lm))

    # Check for valid data
    if np.all(amplitude == 0):
        raise ValueError(f"No valid waveform data for eta={eta}, chi1z={chi1z}, chi2z={chi2z}")

    # Log amplitude (handle zeros at edges)
    with np.errstate(divide='ignore'):
        log_amplitude = np.log10(amplitude)
    log_amplitude[amplitude == 0] = np.nan

    # Align phase at reference frequency
    ref_idx = np.argmin(np.abs(Mf_grid - Mf_ref))
    if amplitude[ref_idx] > 0:
        phase = phase - phase[ref_idx]
    else:
        # Find first valid point if reference is outside valid range
        valid_idx = np.where(amplitude > 0)[0]
        if len(valid_idx) > 0:
            phase = phase - phase[valid_idx[0]]

    # Set phase to NaN where amplitude is zero
    phase[amplitude == 0] = np.nan

    return log_amplitude, phase


# =============================================================================
# Main Data Generation
# =============================================================================

def generate_dataset(
    n_samples: int,
    output_path: str,
    seed: int = 42,
    dataset_name: str = "train",
    ell: int = 2,
    emm: int = 2,
) -> None:
    """
    Generate a dataset of individual mode waveforms.

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
    ell, emm : int
        Spherical harmonic mode numbers.
    """
    print(f"Generating {n_samples} waveforms for {dataset_name} set (mode {ell},{emm})...")

    # Sample parameters
    params = sample_parameters_lhs(n_samples, seed=seed)

    # Create geometric frequency grid (log-spaced)
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
            log_amp, phase = generate_mode_on_geometric_grid(
                eta, chi1z, chi2z, Mf_grid, MF_REF,
                ell=ell, emm=emm,
            )
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
    print(f"Generated {n_success}/{n_samples} waveforms in {elapsed:.1f}s "
          f"({elapsed/n_samples:.3f}s per waveform)")

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
            f.attrs["Mf_ref"] = MF_REF
            f.attrs["n_freq"] = N_FREQ
            f.attrs["frequency_spacing"] = "log"
            f.attrs["approximant"] = "IMRPhenomXHM"
            f.attrs["ell"] = ell
            f.attrs["emm"] = emm
            f.attrs["description"] = (
                f"Aligned spin BBH waveforms, mode ({ell},{emm}). "
                f"Phase aligned at Mf_ref={MF_REF}. "
                f"Log-spaced frequency grid for training. "
                f"See theory/frequency-domain-emulation.tex for framework."
            )

    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate waveform training data for individual modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate (2,2) mode data (default)
    python scripts/generate_data.py

    # Generate (3,3) mode data
    python scripts/generate_data.py --mode 3 3 --output data/waveforms_33mode.h5

    # Quick test
    python scripts/generate_data.py --test
        """
    )
    parser.add_argument("--n_train", type=int, default=10000,
                        help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=1000,
                        help="Number of validation samples")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HDF5 file path (default: data/waveforms_<mode>.h5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--mode", type=int, nargs=2, default=[2, 2],
                        metavar=("ELL", "EMM"),
                        help="Spherical harmonic mode (l, m), default: 2 2")
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode with 100 train, 10 val")
    args = parser.parse_args()

    ell, emm = args.mode

    # Set default output path based on mode
    if args.output is None:
        args.output = f"data/waveforms_{ell}{emm}mode.h5"

    if args.test:
        args.n_train = 100
        args.n_val = 10
        args.output = f"data/waveforms_{ell}{emm}mode_test.h5"

    # Generate training set
    generate_dataset(
        n_samples=args.n_train,
        output_path=args.output,
        seed=args.seed,
        dataset_name="train",
        ell=ell,
        emm=emm,
    )

    # Generate validation set (different seed)
    generate_dataset(
        n_samples=args.n_val,
        output_path=args.output,
        seed=args.seed + 1000,
        dataset_name="validation",
        ell=ell,
        emm=emm,
    )

    # Print summary
    with h5py.File(args.output, "r") as f:
        print("\n=== Dataset Summary ===")
        print(f"File: {args.output}")
        print(f"Mode: ({ell}, {emm})")
        print(f"Frequency grid: {f['frequency_grid'].shape[0]} points (log-spaced)")
        print(f"  Mf range: [{f.attrs['Mf_min']:.4f}, {f.attrs['Mf_max']:.2f}]")
        print(f"  Reference Mf: {f.attrs['Mf_ref']:.4f}")
        print(f"Training samples: {f['train/parameters'].shape[0]}")
        print(f"Validation samples: {f['validation/parameters'].shape[0]}")
        print(f"Parameter bounds:")
        for name in ["eta", "chi1z", "chi2z"]:
            bounds = f.attrs[f"param_bounds_{name}"]
            print(f"  {name}: [{bounds[0]:.2f}, {bounds[1]:.2f}]")


if __name__ == "__main__":
    main()
