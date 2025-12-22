#!/usr/bin/env python
"""Debug phase convention between emulator and LAL."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
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

# Load emulator
with open('outputs/emulator_22mode.pkl', 'rb') as f:
    d = pickle.load(f)

# Reconstruct components
amp_normalizer = PerFrequencyNormalizer.from_dict(d['amp_normalizer'])
phase_normalizer = PerFrequencyNormalizer.from_dict(d['phase_normalizer'])
pca_amplitude = WaveformPCA.from_dict(d['pca_amplitude'])
pca_phase = WaveformPCA.from_dict(d['pca_phase'])
amp_coeff_std = CoefficientStandardizer.from_dict(d['amp_coeff_std'])
phase_coeff_std = CoefficientStandardizer.from_dict(d['phase_coeff_std'])
input_normalizer = InputNormalizer.from_dict(d['input_normalizer'])
Mf_grid = d['frequency_grid']

amp_model = EmulatorMLP(n_hidden=d['n_hidden'], n_units=d['n_units'], n_outputs=d['amp_n_outputs'])
phase_model = EmulatorMLP(n_hidden=d['n_hidden'], n_units=d['n_units'], n_outputs=d['phase_n_outputs'])

# Test parameters
eta = 0.20
chi1z = 0.3
chi2z = -0.2
M_total = 50.0

# Emulator prediction
params = np.array([[eta, chi1z, chi2z]])
params_norm = input_normalizer.transform(params)
params_jax = jnp.array(params_norm)

amp_coeffs_std = amp_model.apply(d['amp_params'], params_jax)
phase_coeffs_std = phase_model.apply(d['phase_params'], params_jax)

amp_coeffs = amp_coeff_std.inverse_transform(np.array(amp_coeffs_std))
phase_coeffs = phase_coeff_std.inverse_transform(np.array(phase_coeffs_std))

amp_norm = pca_amplitude.inverse_transform(amp_coeffs)
phase_norm = pca_phase.inverse_transform(phase_coeffs)

log_amp_pred = amp_normalizer.inverse_transform(amp_norm)[0]
phase_pred = phase_normalizer.inverse_transform(phase_norm)[0]

amp_pred = 10.0 ** log_amp_pred

# Generate LAL waveform
sqrt_term = np.sqrt(1 - 4 * eta)
q = (1 - sqrt_term) / (1 + sqrt_term)
m1 = M_total / (1 + q)
m2 = M_total * q / (1 + q)

f_physical = np.array(geometric_to_physical_frequency(Mf_grid, M_total))
Mf_ref = 0.006
f_ref = float(geometric_to_physical_frequency(np.array([Mf_ref]), M_total)[0])

h_lal = generate_fd_mode_at_frequencies(
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

# Extract LAL amplitude and phase
amp_lal = np.abs(h_lal)
phase_lal = np.unwrap(np.angle(h_lal))

# Align phases at reference
ref_idx = np.argmin(np.abs(Mf_grid - Mf_ref))
phase_lal_aligned = phase_lal - phase_lal[ref_idx]

print("Phase comparison at reference frequency:")
print(f"  Mf_ref = {Mf_grid[ref_idx]:.6f}")
print(f"  phase_pred[ref] = {phase_pred[ref_idx]:.4f}")
print(f"  phase_lal[ref] (before alignment) = {phase_lal[ref_idx]:.4f}")
print(f"  phase_lal[ref] (after alignment) = {phase_lal_aligned[ref_idx]:.4f}")

print("\nPhase comparison at a few frequencies:")
for idx in [0, len(Mf_grid)//4, len(Mf_grid)//2, 3*len(Mf_grid)//4, -1]:
    print(f"  Mf={Mf_grid[idx]:.4f}: pred={phase_pred[idx]:.2f}, lal={phase_lal_aligned[idx]:.2f}, diff={phase_pred[idx]-phase_lal_aligned[idx]:.2f}")

print("\nAmplitude comparison:")
print(f"  amp_pred range: [{amp_pred.min():.2e}, {amp_pred.max():.2e}]")
print(f"  amp_lal range: [{amp_lal.min():.2e}, {amp_lal.max():.2e}]")

# Try different h constructions
print("\n\nTrying different phase conventions...")

# Convention 1: h = A * exp(-i * phi)
h1 = amp_pred * np.exp(-1j * phase_pred)
overlap1 = np.abs(np.sum(np.conj(h_lal) * h1)) / (np.linalg.norm(h_lal) * np.linalg.norm(h1))
print(f"h = A * exp(-i * phase_pred): overlap = {overlap1:.6f}")

# Convention 2: h = A * exp(+i * phi)
h2 = amp_pred * np.exp(1j * phase_pred)
overlap2 = np.abs(np.sum(np.conj(h_lal) * h2)) / (np.linalg.norm(h_lal) * np.linalg.norm(h2))
print(f"h = A * exp(+i * phase_pred): overlap = {overlap2:.6f}")

# Convention 3: Use LAL phase with pred amplitude
h3 = amp_pred * np.exp(1j * phase_lal)
overlap3 = np.abs(np.sum(np.conj(h_lal) * h3)) / (np.linalg.norm(h_lal) * np.linalg.norm(h3))
print(f"h = amp_pred * exp(i * phase_lal): overlap = {overlap3:.6f}")

# Convention 4: Use pred phase aligned to match LAL
# The stored phase might need a global offset
phase_diff = phase_lal_aligned - phase_pred
mean_diff = np.mean(phase_diff)
print(f"\nMean phase difference: {mean_diff:.2f} rad")

h4 = amp_pred * np.exp(1j * (phase_pred + mean_diff))
overlap4 = np.abs(np.sum(np.conj(h_lal) * h4)) / (np.linalg.norm(h_lal) * np.linalg.norm(h4))
print(f"h = A * exp(i * (phase_pred + {mean_diff:.2f})): overlap = {overlap4:.6f}")

# Check if phase_pred matches -phase_lal (opposite sign convention)
phase_diff_neg = -phase_lal_aligned - phase_pred
mean_diff_neg = np.mean(phase_diff_neg)
print(f"\nMean difference with -phase_lal: {mean_diff_neg:.2f} rad")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.loglog(Mf_grid, amp_lal, 'b-', label='LAL')
ax.loglog(Mf_grid, amp_pred, 'r--', label='Emulator')
ax.set_xlabel('Mf')
ax.set_ylabel('Amplitude')
ax.legend()
ax.set_title('Amplitude Comparison')

ax = axes[0, 1]
ax.semilogx(Mf_grid, phase_lal_aligned, 'b-', label='LAL (aligned)')
ax.semilogx(Mf_grid, phase_pred, 'r--', label='Emulator')
ax.semilogx(Mf_grid, -phase_pred, 'g:', label='-Emulator')
ax.set_xlabel('Mf')
ax.set_ylabel('Phase [rad]')
ax.legend()
ax.set_title('Phase Comparison')

ax = axes[1, 0]
ax.semilogx(Mf_grid, phase_lal_aligned - phase_pred, 'b-', label='LAL - pred')
ax.semilogx(Mf_grid, phase_lal_aligned + phase_pred, 'r--', label='LAL + pred')
ax.set_xlabel('Mf')
ax.set_ylabel('Phase difference [rad]')
ax.legend()
ax.set_title('Phase Differences')
ax.axhline(0, color='k', linestyle=':')

ax = axes[1, 1]
ax.semilogx(Mf_grid, (phase_lal_aligned - phase_pred) / (2*np.pi), 'b-')
ax.set_xlabel('Mf')
ax.set_ylabel('Phase difference [cycles]')
ax.set_title('Phase Difference in Cycles')
ax.axhline(0, color='k', linestyle=':')

plt.tight_layout()
plt.savefig('outputs/phase_debug.png', dpi=150)
print("\nSaved: outputs/phase_debug.png")
