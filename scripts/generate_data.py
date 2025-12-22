#!/usr/bin/env python
"""
Generate training data for gravitational waveform emulator.

Raw LAL output on a linear Mf grid - no interpolation, no manipulation.

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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.waveforms import WaveformParameters, generate_fd_waveform
from jim_emulators.waveforms.utils import geometric_to_physical_frequency

# =============================================================================
# Configuration
# =============================================================================

PARAM_BOUNDS = {
    "eta": (0.05, 0.25),
    "chi1z": (-0.99, 0.99),
    "chi2z": (-0.99, 0.99),
}

# Linear Mf grid
MF_MIN = 0.005   # ~20 Hz for M=50 Msun - above waveform start
MF_MAX = 0.25    # ~1015 Hz for M=50 Msun
N_FREQ = 1000

# Reference mass (fixed for all waveforms)
M_REF = 50.0

MODE_22_ONLY = [(2, 2), (2, -2)]


# =============================================================================
# Helpers
# =============================================================================

def sample_parameters_lhs(n_samples: int, seed: int = 42) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    samples = sampler.random(n_samples)
    bounds = np.array([PARAM_BOUNDS["eta"], PARAM_BOUNDS["chi1z"], PARAM_BOUNDS["chi2z"]])
    return qmc.scale(samples, bounds[:, 0], bounds[:, 1]).astype(np.float64)


def eta_to_masses(eta: float, M_total: float) -> Tuple[float, float]:
    sqrt_term = np.sqrt(1 - 4 * eta)
    q = (1 - sqrt_term) / (1 + sqrt_term)
    return M_total / (1 + q), M_total * q / (1 + q)


# =============================================================================
# Waveform Generation
# =============================================================================

def generate_waveform(eta: float, chi1z: float, chi2z: float, f_min: float, f_max: float, delta_f: float):
    """Generate waveform - raw LAL output, sliced to f >= f_min."""
    m1, m2 = eta_to_masses(eta, M_REF)

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

    freqs, hp, _ = generate_fd_waveform(params, mode_array=MODE_22_ONLY, disable_multibanding=True)

    # LAL returns data from f=0, slice to keep only f >= f_min
    valid_mask = freqs >= f_min
    freqs = freqs[valid_mask]
    hp = hp[valid_mask]

    # Raw output
    log_amp = np.log10(np.maximum(np.abs(hp), 1e-100))
    phase = np.unwrap(np.angle(hp))

    return freqs, log_amp, phase


# =============================================================================
# Main
# =============================================================================

def generate_dataset(n_samples: int, output_path: str, seed: int, dataset_name: str, f_min: float, f_max: float, delta_f: float):
    print(f"Generating {n_samples} waveforms for {dataset_name}...")

    params = sample_parameters_lhs(n_samples, seed=seed)

    # Get grid size from first waveform
    freqs, _, _ = generate_waveform(params[0, 0], params[0, 1], params[0, 2], f_min, f_max, delta_f)
    n_freq = len(freqs)
    print(f"  Frequency grid: {n_freq} points, {freqs[0]:.2f} to {freqs[-1]:.2f} Hz")

    amplitudes = np.zeros((n_samples, n_freq), dtype=np.float64)
    phases = np.zeros((n_samples, n_freq), dtype=np.float64)
    success_mask = np.zeros(n_samples, dtype=bool)

    start_time = time.time()
    for i in tqdm(range(n_samples), desc=dataset_name):
        try:
            _, log_amp, phase = generate_waveform(params[i, 0], params[i, 1], params[i, 2], f_min, f_max, delta_f)
            amplitudes[i] = log_amp
            phases[i] = phase
            success_mask[i] = True
        except Exception as e:
            print(f"\nFailed sample {i}: {e}")

    elapsed = time.time() - start_time
    print(f"  Generated {success_mask.sum()}/{n_samples} in {elapsed:.1f}s")

    # Filter successful
    params = params[success_mask]
    amplitudes = amplitudes[success_mask]
    phases = phases[success_mask]

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mode = "a" if os.path.exists(output_path) else "w"

    with h5py.File(output_path, mode) as f:
        if dataset_name in f:
            del f[dataset_name]
        grp = f.create_group(dataset_name)
        grp.create_dataset("parameters", data=params, compression="gzip")
        grp.create_dataset("log_amplitude", data=amplitudes, compression="gzip")
        grp.create_dataset("phase", data=phases, compression="gzip")

        if "freqs" not in f:
            f.create_dataset("freqs", data=freqs)
            f.attrs["f_min"] = f_min
            f.attrs["f_max"] = f_max
            f.attrs["delta_f"] = delta_f
            f.attrs["M_ref"] = M_REF
            f.attrs["Mf_min"] = MF_MIN
            f.attrs["Mf_max"] = MF_MAX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/waveforms_22mode.h5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        args.n_train, args.n_val = 100, 10
        args.output = "data/waveforms_22mode_test.h5"

    # Convert linear Mf grid to physical f grid
    f_min = float(geometric_to_physical_frequency(MF_MIN, M_REF))
    f_max = float(geometric_to_physical_frequency(MF_MAX, M_REF))
    delta_f = (f_max - f_min) / (N_FREQ - 1)

    print(f"Mf grid: [{MF_MIN}, {MF_MAX}] -> f grid: [{f_min:.2f}, {f_max:.2f}] Hz, delta_f={delta_f:.4f}")

    generate_dataset(args.n_train, args.output, args.seed, "train", f_min, f_max, delta_f)
    generate_dataset(args.n_val, args.output, args.seed + 1000, "validation", f_min, f_max, delta_f)

    with h5py.File(args.output, "r") as f:
        print(f"\n=== Summary ===")
        print(f"Train: {f['train/parameters'].shape[0]}, Val: {f['validation/parameters'].shape[0]}")
        print(f"Frequencies: {f['freqs'].shape[0]} points")


if __name__ == "__main__":
    main()
