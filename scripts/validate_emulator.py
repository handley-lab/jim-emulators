#!/usr/bin/env python
"""
Validate the trained emulator against LAL waveforms.

Generates time-domain waveform comparisons and computes mismatch.

Usage:
    python scripts/validate_emulator.py
    python scripts/validate_emulator.py --eta 0.2 --chi1z 0.3 --chi2z -0.2 --mass 50
"""

import argparse
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


def load_emulator(path: str):
    """Load trained emulator from pickle file."""
    with open(path, 'rb') as f:
        d = pickle.load(f)

    # Reconstruct components
    components = {
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
    return components


def predict_waveform(components, eta, chi1z, chi2z):
    """Predict frequency-domain waveform from emulator."""
    params = np.array([[eta, chi1z, chi2z]])
    params_norm = components['input_normalizer'].transform(params)
    params_jax = jnp.array(params_norm)

    # Forward pass
    amp_coeffs_std = components['amp_model'].apply(components['amp_params'], params_jax)
    phase_coeffs_std = components['phase_model'].apply(components['phase_params'], params_jax)

    # Denormalize
    amp_coeffs = components['amp_coeff_std'].inverse_transform(np.array(amp_coeffs_std))
    phase_coeffs = components['phase_coeff_std'].inverse_transform(np.array(phase_coeffs_std))

    # PCA reconstruction
    amp_norm = components['pca_amplitude'].inverse_transform(amp_coeffs)
    phase_norm = components['pca_phase'].inverse_transform(phase_coeffs)

    # Denormalize waveform
    log_amp = components['amp_normalizer'].inverse_transform(amp_norm)[0]
    phase = components['phase_normalizer'].inverse_transform(phase_norm)[0]

    # Convert to complex strain
    # Note: LAL convention is h = A * exp(+i * phase)
    amp = 10.0 ** log_amp
    h = amp * np.exp(1j * phase)

    return h, log_amp, phase


def generate_lal_waveform(Mf_grid, eta, chi1z, chi2z, M_total):
    """Generate LAL waveform for comparison."""
    # Convert eta to masses
    sqrt_term = np.sqrt(1 - 4 * eta)
    q = (1 - sqrt_term) / (1 + sqrt_term)
    m1 = M_total / (1 + q)
    m2 = M_total * q / (1 + q)

    # Convert Mf to physical frequency
    f_physical = np.array(geometric_to_physical_frequency(Mf_grid, M_total))

    # Reference frequency
    Mf_ref = 0.006
    f_ref = float(geometric_to_physical_frequency(np.array([Mf_ref]), M_total)[0])

    h = generate_fd_mode_at_frequencies(
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

    return h, f_physical


def compute_mismatch(h1, h2):
    """Compute simple (unweighted) mismatch between two waveforms.

    Maximizes over overall phase shift (uses |<h1|h2>| not Re<h1|h2>).
    """
    inner = np.abs(np.sum(np.conj(h1) * h2))
    norm1 = np.sqrt(np.sum(np.abs(h1)**2))
    norm2 = np.sqrt(np.sum(np.abs(h2)**2))
    overlap = inner / (norm1 * norm2)
    mismatch = 1 - overlap
    return mismatch, overlap


def main():
    parser = argparse.ArgumentParser(description="Validate emulator against LAL")
    parser.add_argument("--emulator", type=str, default="outputs/emulator_22mode.pkl",
                        help="Path to trained emulator")
    parser.add_argument("--eta", type=float, default=0.20,
                        help="Symmetric mass ratio")
    parser.add_argument("--chi1z", type=float, default=0.3,
                        help="Primary spin")
    parser.add_argument("--chi2z", type=float, default=-0.2,
                        help="Secondary spin")
    parser.add_argument("--mass", type=float, default=50.0,
                        help="Total mass in solar masses")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading emulator...")
    components = load_emulator(args.emulator)
    Mf_grid = components['Mf_grid']

    print(f"Test parameters: η={args.eta}, χ₁z={args.chi1z}, χ₂z={args.chi2z}, M={args.mass} M☉")

    # Emulator prediction
    print("Generating emulator waveform...")
    h_pred, log_amp_pred, phase_pred = predict_waveform(
        components, args.eta, args.chi1z, args.chi2z
    )

    # LAL waveform
    print("Generating LAL waveform...")
    h_true, f_physical = generate_lal_waveform(
        Mf_grid, args.eta, args.chi1z, args.chi2z, args.mass
    )

    # Compute mismatch
    mismatch, overlap = compute_mismatch(h_true, h_pred)
    print(f"\nSimple (unweighted) overlap: {overlap:.6f}")
    print(f"Simple mismatch: {mismatch:.2e}")

    # Time domain via IFFT
    n_freq = len(Mf_grid)
    n_fft = 2 ** int(np.ceil(np.log2(2 * n_freq)))

    h_pred_full = np.zeros(n_fft, dtype=complex)
    h_true_full = np.zeros(n_fft, dtype=complex)

    h_pred_full[1:n_freq+1] = h_pred
    h_true_full[1:n_freq+1] = h_true

    # Conjugate symmetry
    h_pred_full[n_fft-n_freq+1:] = np.conj(h_pred[::-1][:-1])
    h_true_full[n_fft-n_freq+1:] = np.conj(h_true[::-1][:-1])

    h_pred_td = np.fft.ifft(h_pred_full) * n_fft
    h_true_td = np.fft.ifft(h_true_full) * n_fft

    # Time array
    dt = 1.0 / (2 * f_physical[-1])

    # Find merger
    peak_idx_pred = np.argmax(np.abs(h_pred_td))
    peak_idx_true = np.argmax(np.abs(h_true_td))

    t_centered_pred = (np.arange(n_fft) - peak_idx_pred) * dt * 1000  # ms
    t_centered_true = (np.arange(n_fft) - peak_idx_true) * dt * 1000  # ms

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Frequency domain amplitude
    ax = axes[0, 0]
    ax.loglog(Mf_grid, np.abs(h_true), 'b-', label='LAL', linewidth=1.5)
    ax.loglog(Mf_grid, np.abs(h_pred), 'r--', label='Emulator', linewidth=1.5)
    ax.set_xlabel('Mf (geometric frequency)')
    ax.set_ylabel('|h|')
    ax.set_title('Frequency Domain Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Frequency domain phase
    ax = axes[0, 1]
    phase_true = np.unwrap(np.angle(h_true))
    Mf_ref = 0.006
    ref_idx = np.argmin(np.abs(Mf_grid - Mf_ref))
    phase_true = phase_true - phase_true[ref_idx]
    ax.semilogx(Mf_grid, phase_true, 'b-', label='LAL', linewidth=1.5)
    ax.semilogx(Mf_grid, phase_pred, 'r--', label='Emulator', linewidth=1.5)
    ax.set_xlabel('Mf (geometric frequency)')
    ax.set_ylabel('Phase [rad]')
    ax.set_title('Frequency Domain Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Time domain
    ax = axes[1, 0]
    window = 50  # ms
    mask_true = np.abs(t_centered_true) < window
    mask_pred = np.abs(t_centered_pred) < window
    ax.plot(t_centered_true[mask_true], np.real(h_true_td)[mask_true], 'b-', label='LAL', linewidth=1)
    ax.plot(t_centered_pred[mask_pred], np.real(h_pred_td)[mask_pred], 'r--', label='Emulator', linewidth=1)
    ax.set_xlabel('Time from merger [ms]')
    ax.set_ylabel('h(t)')
    ax.set_title(f'Time Domain Waveform (M={args.mass} M☉)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-window, window)

    # Residual
    ax = axes[1, 1]
    shift = peak_idx_true - peak_idx_pred
    h_pred_aligned = np.roll(h_pred_td, shift)
    residual = np.real(h_true_td - h_pred_aligned)
    ax.plot(t_centered_true[mask_true], residual[mask_true], 'g-', linewidth=1)
    ax.set_xlabel('Time from merger [ms]')
    ax.set_ylabel('Residual h_true - h_pred')
    ax.set_title('Time Domain Residual')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-window, window)

    fig.suptitle(
        f'Emulator vs LAL: η={args.eta}, χ₁z={args.chi1z}, χ₂z={args.chi2z}, M={args.mass} M☉\n'
        f'Mismatch: {mismatch:.2e}',
        fontsize=12
    )

    plt.tight_layout()
    plot_path = output_dir / "time_domain_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved: {plot_path}")


if __name__ == "__main__":
    main()
