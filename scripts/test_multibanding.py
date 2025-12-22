#!/usr/bin/env python
"""
Test the effect of disabling multibanding on waveform smoothness.

Multibanding is LAL's optimization that computes waveforms on a coarser grid
and interpolates. Disabling it should produce smoother waveforms, especially
at low amplitudes.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "src")

from jim_emulators.waveforms import (
    WaveformParameters,
    generate_fd_waveform,
)


def compare_multibanding():
    """Compare waveforms with and without multibanding."""

    # Test parameters - use a case where higher modes are significant
    params = WaveformParameters(
        mass_1=35.0,
        mass_2=25.0,
        chi1z=0.3,
        chi2z=-0.2,
        luminosity_distance=100.0,
        inclination=np.pi / 3,  # Edge-on to see higher modes
        f_min=20.0,
        f_max=512.0,
        delta_f=0.125,
        approximant="IMRPhenomXPHM",
    )

    print("=" * 70)
    print("Testing Multibanding Effect on Waveform Smoothness")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  m1 = {params.mass_1} M_sun, m2 = {params.mass_2} M_sun")
    print(f"  chi1z = {params.chi1z}, chi2z = {params.chi2z}")
    print(f"  inclination = {params.inclination:.4f} rad")

    # Generate with default multibanding
    print("\n1. Generating with multibanding (default)...")
    freqs_mb, hp_mb, hc_mb = generate_fd_waveform(params, disable_multibanding=False)

    # Generate without multibanding
    print("2. Generating without multibanding (exact)...")
    freqs_no_mb, hp_no_mb, hc_no_mb = generate_fd_waveform(params, disable_multibanding=True)

    # Compare
    print("\n3. Comparing results...")

    mask = (freqs_mb >= params.f_min) & (freqs_mb <= params.f_max)

    diff = np.abs(hp_mb[mask] - hp_no_mb[mask])
    max_amp = np.max(np.abs(hp_mb[mask]))

    print(f"\nMax |h_multibanding - h_exact|: {np.max(diff):.3e}")
    print(f"Max relative difference: {np.max(diff) / max_amp:.3e}")

    # Compute match
    def compute_match(h1, h2, freqs, f_low=20.0):
        mask = freqs >= f_low
        h1_m, h2_m = h1[mask], h2[mask]
        df = freqs[1] - freqs[0]
        inner_11 = 4 * df * np.sum(np.abs(h1_m)**2).real
        inner_22 = 4 * df * np.sum(np.abs(h2_m)**2).real
        inner_12 = 4 * df * np.sum(h1_m * np.conj(h2_m))
        return np.abs(inner_12) / np.sqrt(inner_11 * inner_22)

    match = compute_match(hp_mb, hp_no_mb, freqs_mb)
    print(f"Match: {match:.15f}")
    print(f"Mismatch: {1 - match:.3e}")

    # Create comparison plots
    print("\n4. Generating comparison plots...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Effect of Disabling Multibanding on IMRPhenomXPHM", fontsize=14)

    freqs_plot = freqs_mb[mask]
    amp_mb = np.abs(hp_mb[mask])
    amp_no_mb = np.abs(hp_no_mb[mask])
    phase_mb = np.unwrap(np.angle(hp_mb[mask]))
    phase_no_mb = np.unwrap(np.angle(hp_no_mb[mask]))

    # Amplitude comparison
    ax = axes[0, 0]
    ax.semilogy(freqs_plot, amp_mb, 'b-', label='Multibanding ON', linewidth=1.5, alpha=0.8)
    ax.semilogy(freqs_plot, amp_no_mb, 'r--', label='Multibanding OFF', linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|h+(f)|")
    ax.set_title("Amplitude Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Amplitude difference
    ax = axes[0, 1]
    ax.semilogy(freqs_plot, np.abs(amp_mb - amp_no_mb) + 1e-50, 'g-', linewidth=1)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|Amp_MB - Amp_exact|")
    ax.set_title("Amplitude Difference")
    ax.grid(True, alpha=0.3)

    # Phase comparison
    ax = axes[1, 0]
    ax.plot(freqs_plot, phase_mb, 'b-', label='Multibanding ON', linewidth=1.5, alpha=0.8)
    ax.plot(freqs_plot, phase_no_mb, 'r--', label='Multibanding OFF', linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Phase [rad]")
    ax.set_title("Phase Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phase difference
    ax = axes[1, 1]
    phase_diff = phase_mb - phase_no_mb
    ax.plot(freqs_plot, phase_diff, 'g-', linewidth=1)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Phase_MB - Phase_exact [rad]")
    ax.set_title("Phase Difference")
    ax.grid(True, alpha=0.3)

    # Zoomed amplitude at high frequency (where differences are most visible)
    ax = axes[2, 0]
    zoom_mask = (freqs_plot >= 200) & (freqs_plot <= 400)
    ax.semilogy(freqs_plot[zoom_mask], amp_mb[zoom_mask], 'b-', label='MB ON', linewidth=1.5)
    ax.semilogy(freqs_plot[zoom_mask], amp_no_mb[zoom_mask], 'r--', label='MB OFF', linewidth=1.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|h+(f)|")
    ax.set_title("Amplitude Zoom (200-400 Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative difference
    ax = axes[2, 1]
    rel_diff = np.abs(amp_mb - amp_no_mb) / (amp_no_mb + 1e-50)
    ax.semilogy(freqs_plot, rel_diff + 1e-16, 'g-', linewidth=1)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Relative difference")
    ax.set_title("Relative Amplitude Difference")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/multibanding_comparison.png", dpi=150)
    print(f"\nSaved: figures/multibanding_comparison.png")
    plt.close()


def compare_modes_with_multibanding():
    """Compare individual modes with and without multibanding."""

    params = WaveformParameters(
        mass_1=35.0,
        mass_2=25.0,
        chi1z=0.3,
        chi2z=-0.2,
        luminosity_distance=100.0,
        inclination=np.pi / 3,
        f_min=20.0,
        f_max=512.0,
        delta_f=0.125,
        approximant="IMRPhenomXPHM",
    )

    modes = [(2, 2), (3, 3), (4, 4)]

    print("\n" + "=" * 70)
    print("Comparing Individual Modes With/Without Multibanding")
    print("=" * 70)

    fig, axes = plt.subplots(len(modes), 2, figsize=(14, 4 * len(modes)))
    fig.suptitle("Mode-by-Mode Multibanding Comparison", fontsize=14, y=1.02)

    for idx, mode in enumerate(modes):
        print(f"\nMode {mode}...")

        # With multibanding
        freqs, hp_mb, _ = generate_fd_waveform(
            params, mode_array=[mode], disable_multibanding=False
        )

        # Without multibanding
        _, hp_no_mb, _ = generate_fd_waveform(
            params, mode_array=[mode], disable_multibanding=True
        )

        mask = (freqs >= params.f_min) & (freqs <= params.f_max)
        freqs_plot = freqs[mask]

        amp_mb = np.abs(hp_mb[mask])
        amp_no_mb = np.abs(hp_no_mb[mask])

        # Skip if mode has negligible amplitude
        if np.max(amp_no_mb) < 1e-40:
            print(f"  Skipping - negligible amplitude")
            continue

        diff = np.abs(amp_mb - amp_no_mb)
        rel_diff = diff / (amp_no_mb + 1e-50)

        print(f"  Max |diff|: {np.max(diff):.3e}")
        print(f"  Max relative diff: {np.max(rel_diff):.3e}")

        # Amplitude comparison
        ax = axes[idx, 0]
        ax.semilogy(freqs_plot, amp_mb, 'b-', label='MB ON', linewidth=1.5, alpha=0.8)
        ax.semilogy(freqs_plot, amp_no_mb, 'r--', label='MB OFF', linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("|h+(f)|")
        ax.set_title(f"Mode ({mode[0]}, ±{mode[1]}): Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Relative difference
        ax = axes[idx, 1]
        ax.semilogy(freqs_plot, rel_diff + 1e-16, 'g-', linewidth=1)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Relative difference")
        ax.set_title(f"Mode ({mode[0]}, ±{mode[1]}): Relative Difference")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/multibanding_modes_comparison.png", dpi=150)
    print(f"\nSaved: figures/multibanding_modes_comparison.png")
    plt.close()


def test_smoothness_improvement():
    """
    Generate sensitivity plots with and without multibanding to compare smoothness.
    """
    import jax.numpy as jnp
    from jim_emulators.waveforms.utils import fd_to_td

    params_base = {
        "mass_2": 25.0,
        "chi1z": 0.3,
        "chi2z": -0.2,
        "luminosity_distance": 100.0,
        "inclination": np.pi / 3,
        "f_min": 20.0,
        "f_max": 512.0,
        "delta_f": 0.125,
    }

    # Vary mass
    mass_values = np.linspace(25, 50, 20)

    print("\n" + "=" * 70)
    print("Sensitivity Comparison: Multibanding ON vs OFF")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Sensitivity to Mass: Multibanding Effect on Smoothness", fontsize=14)

    for mb_idx, (disable_mb, label, color) in enumerate([
        (False, "Multibanding ON", "viridis"),
        (True, "Multibanding OFF", "plasma"),
    ]):
        print(f"\nGenerating with {label}...")

        hp_list = []
        for m1 in mass_values:
            params = WaveformParameters(
                mass_1=m1,
                **params_base,
                approximant="IMRPhenomXPHM",
            )
            freqs, hp, _ = generate_fd_waveform(params, disable_multibanding=disable_mb)
            hp_list.append(hp)

        hp_grid = np.array(hp_list)
        mask = (freqs >= 20) & (freqs <= 300)
        freqs_plot = freqs[mask]
        delta_f = freqs[1] - freqs[0]

        from matplotlib.colors import Normalize
        from matplotlib import cm
        norm = Normalize(vmin=mass_values.min(), vmax=mass_values.max())
        cmap = cm.get_cmap(color)

        # Log amplitude
        ax = axes[mb_idx, 0]
        for hp, m1 in zip(hp_grid, mass_values):
            ax.plot(freqs_plot, np.log10(np.abs(hp[mask]) + 1e-50),
                   color=cmap(norm(m1)), alpha=0.7, linewidth=0.8)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("$\\log_{10}|h_+(f)|$")
        ax.set_title(f"{label}: Log Amplitude")
        ax.grid(True, alpha=0.3)

        # Phase
        ax = axes[mb_idx, 1]
        for hp, m1 in zip(hp_grid, mass_values):
            phase = np.unwrap(np.angle(hp[mask]))
            ax.plot(freqs_plot, phase, color=cmap(norm(m1)), alpha=0.7, linewidth=0.8)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Phase [rad]")
        ax.set_title(f"{label}: Unwrapped Phase")
        ax.grid(True, alpha=0.3)

        # Time domain
        ax = axes[mb_idx, 2]
        for hp, m1 in zip(hp_grid, mass_values):
            t, h_td = fd_to_td(jnp.array(hp), delta_f, center=True)
            t = np.array(t)
            h_td = np.array(h_td)
            t_mask = np.abs(t) < 0.1
            ax.plot(t[t_mask] * 1000, h_td[t_mask],
                   color=cmap(norm(m1)), alpha=0.7, linewidth=0.8)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("h(t)")
        ax.set_title(f"{label}: Time Domain")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/multibanding_smoothness_comparison.png", dpi=150)
    print(f"\nSaved: figures/multibanding_smoothness_comparison.png")
    plt.close()


if __name__ == "__main__":
    compare_multibanding()
    compare_modes_with_multibanding()
    test_smoothness_improvement()
