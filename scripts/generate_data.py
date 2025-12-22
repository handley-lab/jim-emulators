#!/usr/bin/env python
"""
Generate training data for gravitational waveform emulator.

Amplitude only, all modes, raw LAL output on a linear Mf grid.

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

# All modes to generate (each with +m and -m)
MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]


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


def mode_to_str(l: int, m: int) -> str:
    """Convert (l, m) to string like '22', '21', '33', etc."""
    return f"{l}{m}"


# =============================================================================
# Waveform Generation
# =============================================================================

def generate_mode_amplitude(
    eta: float, chi1z: float, chi2z: float,
    l: int, m: int,
    f_min: float, f_max: float, delta_f: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate amplitude for a single mode - raw LAL output, sliced to f >= f_min."""
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

    # Generate with just this mode (and its negative-m partner)
    mode_array = [(l, m), (l, -m)]
    freqs, hp, _ = generate_fd_waveform(params, mode_array=mode_array, disable_multibanding=True)

    # LAL returns data from f=0, slice to keep only f >= f_min
    valid_mask = freqs >= f_min
    freqs = freqs[valid_mask]
    hp = hp[valid_mask]

    # Raw log amplitude
    log_amp = np.log10(np.maximum(np.abs(hp), 1e-100))

    return freqs, log_amp


# =============================================================================
# Main
# =============================================================================

def generate_dataset(
    n_samples: int, output_path: str, seed: int, dataset_name: str,
    f_min: float, f_max: float, delta_f: float, modes: list
):
    print(f"\nGenerating {n_samples} waveforms for {dataset_name}...")

    params = sample_parameters_lhs(n_samples, seed=seed)

    # Get grid size from first waveform (using 22 mode)
    freqs, _ = generate_mode_amplitude(params[0, 0], params[0, 1], params[0, 2], 2, 2, f_min, f_max, delta_f)
    n_freq = len(freqs)
    print(f"  Frequency grid: {n_freq} points, {freqs[0]:.2f} to {freqs[-1]:.2f} Hz")

    # Storage for each mode
    mode_amplitudes = {mode_to_str(l, m): np.zeros((n_samples, n_freq), dtype=np.float64) for l, m in modes}
    success_mask = np.ones(n_samples, dtype=bool)

    start_time = time.time()
    for i in tqdm(range(n_samples), desc=dataset_name):
        eta, chi1z, chi2z = params[i]
        for l, m in modes:
            try:
                _, log_amp = generate_mode_amplitude(eta, chi1z, chi2z, l, m, f_min, f_max, delta_f)
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

        # Store amplitude for each mode
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/waveforms_multimode.h5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        args.n_train, args.n_val = 100, 10
        args.output = "data/waveforms_multimode_test.h5"

    # Convert linear Mf grid to physical f grid
    f_min = float(geometric_to_physical_frequency(MF_MIN, M_REF))
    f_max = float(geometric_to_physical_frequency(MF_MAX, M_REF))
    delta_f = (f_max - f_min) / (N_FREQ - 1)

    print(f"Mf grid: [{MF_MIN}, {MF_MAX}] -> f grid: [{f_min:.2f}, {f_max:.2f}] Hz, delta_f={delta_f:.4f}")
    print(f"Modes: {MODES}")

    generate_dataset(args.n_train, args.output, args.seed, "train", f_min, f_max, delta_f, MODES)
    generate_dataset(args.n_val, args.output, args.seed + 1000, "validation", f_min, f_max, delta_f, MODES)

    with h5py.File(args.output, "r") as f:
        print(f"\n=== Summary ===")
        print(f"Train: {f['train/parameters'].shape[0]}, Val: {f['validation/parameters'].shape[0]}")
        print(f"Frequencies: {f['freqs'].shape[0]} points")
        print(f"Modes: {list(f.attrs['modes'])}")
        for mode in f.attrs['modes']:
            print(f"  {mode}: shape {f[f'train/log_amplitude_{mode}'].shape}")


if __name__ == "__main__":
    main()
