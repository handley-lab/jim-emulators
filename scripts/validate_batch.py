#!/usr/bin/env python
"""Validate emulator on multiple test points."""

import pickle
import numpy as np
import h5py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.emulator import (
    WaveformPCA,
    EmulatorMLP,
    PerFrequencyNormalizer,
    CoefficientStandardizer,
    InputNormalizer,
)
from jim_emulators.waveforms import generate_fd_mode_at_frequencies
from jim_emulators.waveforms.utils import geometric_to_physical_frequency

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def load_emulator(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return {
        'amp_normalizer': PerFrequencyNormalizer.from_dict(d['amp_normalizer']),
        'phase_normalizer': PerFrequencyNormalizer.from_dict(d['phase_normalizer']),
        'pca_amplitude': WaveformPCA.from_dict(d['pca_amplitude']),
        'pca_phase': WaveformPCA.from_dict(d['pca_phase']),
        'amp_coeff_std': CoefficientStandardizer.from_dict(d['amp_coeff_std']),
        'phase_coeff_std': CoefficientStandardizer.from_dict(d['phase_coeff_std']),
        'input_normalizer': InputNormalizer.from_dict(d['input_normalizer']),
        'Mf_grid': d['frequency_grid'],
        'amp_model': EmulatorMLP(n_hidden=d['n_hidden'], n_units=d['n_units'], n_outputs=d['amp_n_outputs']),
        'phase_model': EmulatorMLP(n_hidden=d['n_hidden'], n_units=d['n_units'], n_outputs=d['phase_n_outputs']),
        'amp_params': d['amp_params'],
        'phase_params': d['phase_params'],
    }


def predict_h(components, eta, chi1z, chi2z):
    params = np.array([[eta, chi1z, chi2z]])
    params_norm = components['input_normalizer'].transform(params)
    params_jax = jnp.array(params_norm)

    amp_coeffs_std = components['amp_model'].apply(components['amp_params'], params_jax)
    phase_coeffs_std = components['phase_model'].apply(components['phase_params'], params_jax)

    amp_coeffs = components['amp_coeff_std'].inverse_transform(np.array(amp_coeffs_std))
    phase_coeffs = components['phase_coeff_std'].inverse_transform(np.array(phase_coeffs_std))

    amp_norm = components['pca_amplitude'].inverse_transform(amp_coeffs)
    phase_norm = components['pca_phase'].inverse_transform(phase_coeffs)

    log_amp = components['amp_normalizer'].inverse_transform(amp_norm)[0]
    phase = components['phase_normalizer'].inverse_transform(phase_norm)[0]

    amp = 10.0 ** log_amp
    h = amp * np.exp(1j * phase)
    return h


def generate_lal_h(Mf_grid, eta, chi1z, chi2z, M_total=50.0):
    sqrt_term = np.sqrt(1 - 4 * eta)
    q = (1 - sqrt_term) / (1 + sqrt_term)
    m1 = M_total / (1 + q)
    m2 = M_total * q / (1 + q)

    f_physical = np.array(geometric_to_physical_frequency(Mf_grid, M_total))
    Mf_ref = 0.006
    f_ref = float(geometric_to_physical_frequency(np.array([Mf_ref]), M_total)[0])

    return generate_fd_mode_at_frequencies(
        frequencies=f_physical,
        mass_1=m1,
        mass_2=m2,
        chi1z=chi1z,
        chi2z=chi2z,
        ell=2,
        emm=2,
        luminosity_distance=1.0,
        phase=0.0,
        f_ref=f_ref,
    )


def compute_mismatch(h1, h2):
    inner = np.abs(np.sum(np.conj(h1) * h2))
    norm1 = np.sqrt(np.sum(np.abs(h1)**2))
    norm2 = np.sqrt(np.sum(np.abs(h2)**2))
    overlap = inner / (norm1 * norm2)
    return 1 - overlap


# Load emulator
print("Loading emulator...")
components = load_emulator('outputs/emulator_22mode.pkl')
Mf_grid = components['Mf_grid']

# Load validation data
print("Loading validation data...")
with h5py.File('data/waveforms_22mode.h5', 'r') as f:
    val_params = f['validation/parameters'][:]

print(f"\nValidation set: {len(val_params)} samples")

# Test on random validation samples
np.random.seed(42)
n_test = 20
test_indices = np.random.choice(len(val_params), n_test, replace=False)

print(f"\nTesting on {n_test} validation samples:")
print("-" * 70)
print(f"{'idx':>4}  {'eta':>6}  {'chi1z':>6}  {'chi2z':>6}  {'mismatch':>12}  {'log10(mm)':>10}")
print("-" * 70)

mismatches = []
for idx in test_indices:
    eta, chi1z, chi2z = val_params[idx]

    h_pred = predict_h(components, eta, chi1z, chi2z)
    h_true = generate_lal_h(Mf_grid, eta, chi1z, chi2z)

    mm = compute_mismatch(h_true, h_pred)
    mismatches.append(mm)

    print(f"{idx:4d}  {eta:6.3f}  {chi1z:6.2f}  {chi2z:6.2f}  {mm:12.2e}  {np.log10(mm):10.2f}")

mismatches = np.array(mismatches)
print("-" * 70)
print(f"\nStatistics:")
print(f"  Mean mismatch: {mismatches.mean():.2e}")
print(f"  Median mismatch: {np.median(mismatches):.2e}")
print(f"  Min mismatch: {mismatches.min():.2e}")
print(f"  Max mismatch: {mismatches.max():.2e}")
print(f"  Fraction < 10^-3: {(mismatches < 1e-3).sum()}/{n_test}")
print(f"  Fraction < 10^-2: {(mismatches < 1e-2).sum()}/{n_test}")
