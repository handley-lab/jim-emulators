#!/usr/bin/env python
"""
Generate training data for PRECESSING gravitational waveform emulator.

Full parameter space with precessing spins:
- eta: symmetric mass ratio
- chi1, chi2: spin magnitudes (0 to 0.99)
- tilt1, tilt2: spin tilt angles (0 to pi)
- phi12: azimuthal angle between spins (0 to 2pi)
- phi_jl: azimuthal angle of L around J (0 to 2pi)

This is 7 parameters total, compared to 3 for aligned spins.

Usage:
    python scripts/generate_data_precessing.py --test          # Quick test
    python scripts/generate_data_precessing.py --n_train 50000 # Full dataset
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.waveforms import WaveformParameters, generate_fd_waveform
from jim_emulators.waveforms.utils import geometric_to_physical_frequency

# =============================================================================
# Configuration
# =============================================================================

# Parameter bounds for precessing spins
PARAM_BOUNDS = {
    "eta": (0.05, 0.25),        # Symmetric mass ratio
    "chi1": (0.0, 0.99),        # Spin 1 magnitude
    "chi2": (0.0, 0.99),        # Spin 2 magnitude
    "tilt1": (0.0, np.pi),      # Spin 1 tilt from L
    "tilt2": (0.0, np.pi),      # Spin 2 tilt from L
    "phi12": (0.0, 2 * np.pi),  # Azimuthal angle between spins
    "phi_jl": (0.0, 2 * np.pi), # Azimuthal angle of L around J
}

PARAM_NAMES = ["eta", "chi1", "chi2", "tilt1", "tilt2", "phi12", "phi_jl"]
N_PARAMS = len(PARAM_NAMES)

# Linear Mf grid
MF_MIN = 0.005
MF_MAX = 0.25
N_FREQ = 1000

# Reference mass
M_REF = 50.0

# Modes to generate
MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]


# =============================================================================
# Helpers
# =============================================================================

def sample_parameters_lhs(n_samples: int, seed: int = 42) -> np.ndarray:
    """Sample parameters using Latin Hypercube Sampling."""
    sampler = qmc.LatinHypercube(d=N_PARAMS, seed=seed)
    samples = sampler.random(n_samples)

    bounds = np.array([PARAM_BOUNDS[name] for name in PARAM_NAMES])
    return qmc.scale(samples, bounds[:, 0], bounds[:, 1]).astype(np.float64)


def eta_to_masses(eta: float, M_total: float) -> Tuple[float, float]:
    """Convert symmetric mass ratio to component masses."""
    sqrt_term = np.sqrt(1 - 4 * eta)
    q = (1 - sqrt_term) / (1 + sqrt_term)
    return M_total / (1 + q), M_total * q / (1 + q)


def spherical_to_cartesian_spin(chi: float, tilt: float, phi: float) -> Tuple[float, float, float]:
    """
    Convert spin magnitude and angles to Cartesian components.

    Parameters
    ----------
    chi : float
        Spin magnitude (0 to 1)
    tilt : float
        Tilt angle from orbital angular momentum L (0 to pi)
    phi : float
        Azimuthal angle in the orbital plane (0 to 2pi)

    Returns
    -------
    chi_x, chi_y, chi_z : float
        Cartesian spin components
    """
    chi_x = chi * np.sin(tilt) * np.cos(phi)
    chi_y = chi * np.sin(tilt) * np.sin(phi)
    chi_z = chi * np.cos(tilt)
    return chi_x, chi_y, chi_z


def mode_to_str(l: int, m: int) -> str:
    return f"{l}{m}"


# =============================================================================
# Waveform Generation
# =============================================================================

def generate_mode_amplitude(
    eta: float, chi1: float, chi2: float,
    tilt1: float, tilt2: float, phi12: float, phi_jl: float,
    l: int, m: int,
    f_min: float, f_max: float, delta_f: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate amplitude for a single mode with precessing spins."""
    m1, m2 = eta_to_masses(eta, M_REF)

    # Convert to Cartesian spins
    # For spin 1, use phi_jl as reference
    # For spin 2, offset by phi12
    chi1x, chi1y, chi1z = spherical_to_cartesian_spin(chi1, tilt1, phi_jl)
    chi2x, chi2y, chi2z = spherical_to_cartesian_spin(chi2, tilt2, phi_jl + phi12)

    params = WaveformParameters(
        mass_1=m1,
        mass_2=m2,
        chi1x=chi1x,
        chi1y=chi1y,
        chi1z=chi1z,
        chi2x=chi2x,
        chi2y=chi2y,
        chi2z=chi2z,
        luminosity_distance=1.0,
        inclination=0.0,  # Face-on for amplitude
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


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_dataset(
    n_samples: int, output_path: str, seed: int, dataset_name: str,
    f_min: float, f_max: float, delta_f: float, modes: list
):
    print(f"\nGenerating {n_samples} PRECESSING waveforms for {dataset_name}...")

    params = sample_parameters_lhs(n_samples, seed=seed)

    # Get grid size from first waveform
    p = params[0]
    freqs, _ = generate_mode_amplitude(
        p[0], p[1], p[2], p[3], p[4], p[5], p[6],
        2, 2, f_min, f_max, delta_f
    )
    n_freq = len(freqs)
    print(f"  Frequency grid: {n_freq} points, {freqs[0]:.2f} to {freqs[-1]:.2f} Hz")
    print(f"  Parameter space: {N_PARAMS}D ({', '.join(PARAM_NAMES)})")

    # Storage
    mode_amplitudes = {mode_to_str(l, m): np.zeros((n_samples, n_freq), dtype=np.float64) for l, m in modes}
    success_mask = np.ones(n_samples, dtype=bool)

    start_time = time.time()
    for i in tqdm(range(n_samples), desc=dataset_name):
        eta, chi1, chi2, tilt1, tilt2, phi12, phi_jl = params[i]

        for l, m in modes:
            try:
                _, log_amp = generate_mode_amplitude(
                    eta, chi1, chi2, tilt1, tilt2, phi12, phi_jl,
                    l, m, f_min, f_max, delta_f
                )
                mode_amplitudes[mode_to_str(l, m)][i] = log_amp
            except Exception as e:
                print(f"\nFailed sample {i}, mode ({l},{m}): {e}")
                success_mask[i] = False
                break

    elapsed = time.time() - start_time
    print(f"  Generated {success_mask.sum()}/{n_samples} in {elapsed:.1f}s")

    # Filter successful
    params = params[success_mask]
    for key in mode_amplitudes:
        mode_amplitudes[key] = mode_amplitudes[key][success_mask]

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mode = "a" if os.path.exists(output_path) else "w"

    with h5py.File(output_path, mode) as f:
        if dataset_name in f:
            del f[dataset_name]
        grp = f.create_group(dataset_name)
        grp.create_dataset("parameters", data=params, compression="gzip")

        for mode_str, amplitudes in mode_amplitudes.items():
            grp.create_dataset(f"log_amplitude_{mode_str}", data=amplitudes, compression="gzip")

        if "freqs" not in f:
            f.create_dataset("freqs", data=freqs)
            f.attrs["f_min"] = f_min
            f.attrs["f_max"] = f_max
            f.attrs["delta_f"] = delta_f
            f.attrs["M_ref"] = M_REF
            f.attrs["Mf_min"] = MF_MIN
            f.attrs["Mf_max"] = MF_MAX
            f.attrs["modes"] = [mode_to_str(l, m) for l, m in modes]
            f.attrs["param_names"] = PARAM_NAMES
            f.attrs["n_params"] = N_PARAMS
            f.attrs["precessing"] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=50000,
                        help="Training samples (default 50k for 7D space)")
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--output", type=str, default="data/waveforms_precessing.h5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        args.n_train, args.n_val = 500, 50
        args.output = "data/waveforms_precessing_test.h5"

    # Convert Mf grid to physical f
    f_min = float(geometric_to_physical_frequency(MF_MIN, M_REF))
    f_max = float(geometric_to_physical_frequency(MF_MAX, M_REF))
    delta_f = (f_max - f_min) / (N_FREQ - 1)

    print("=" * 60)
    print("Generating PRECESSING Waveform Training Data")
    print("=" * 60)
    print(f"Mf grid: [{MF_MIN}, {MF_MAX}] -> f grid: [{f_min:.2f}, {f_max:.2f}] Hz")
    print(f"Parameters: {PARAM_NAMES}")
    print(f"Modes: {MODES}")

    generate_dataset(args.n_train, args.output, args.seed, "train", f_min, f_max, delta_f, MODES)
    generate_dataset(args.n_val, args.output, args.seed + 1000, "validation", f_min, f_max, delta_f, MODES)

    with h5py.File(args.output, "r") as f:
        print(f"\n=== Summary ===")
        print(f"Train: {f['train/parameters'].shape[0]} samples, {f['train/parameters'].shape[1]} params")
        print(f"Val: {f['validation/parameters'].shape[0]} samples")
        print(f"Frequencies: {f['freqs'].shape[0]} points")
        print(f"Modes: {list(f.attrs['modes'])}")


if __name__ == "__main__":
    main()
