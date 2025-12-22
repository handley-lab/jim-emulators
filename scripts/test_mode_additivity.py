#!/usr/bin/env python
"""
Test if XPHM modes are additive: h(f) = sum_{lm} h_lm(f)

This tests whether the full waveform equals the sum of individual mode contributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "src")

from jim_emulators.waveforms import (
    WaveformParameters,
    generate_fd_waveform,
    DEFAULT_MODES_XPHM,
)


def test_mode_additivity():
    """Test if XPHM h(f) = sum of individual mode contributions."""

    # Test parameters
    params = WaveformParameters(
        mass_1=35.0,
        mass_2=25.0,
        chi1z=0.3,
        chi2z=-0.2,
        luminosity_distance=100.0,
        inclination=np.pi / 4,  # Non-zero to see all modes
        f_min=20.0,
        f_max=512.0,
        delta_f=0.125,
        approximant="IMRPhenomXPHM",
    )

    print("=" * 70)
    print("Testing Mode Additivity for IMRPhenomXPHM")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  m1 = {params.mass_1} M_sun, m2 = {params.mass_2} M_sun")
    print(f"  chi1z = {params.chi1z}, chi2z = {params.chi2z}")
    print(f"  inclination = {params.inclination:.4f} rad")
    print(f"\nAvailable modes: {DEFAULT_MODES_XPHM}")

    # 1. Generate full waveform (all modes)
    print("\n1. Generating full waveform (all modes)...")
    freqs, hp_full, hc_full = generate_fd_waveform(params, mode_array=None)

    # 2. Generate individual mode contributions
    print("2. Generating individual mode contributions...")

    # Group modes by |m| to handle +m and -m together
    mode_groups = [
        [(2, 2), (2, -2)],   # Dominant quadrupole
        [(2, 1), (2, -1)],   # Subdominant quadrupole
        [(3, 3), (3, -3)],   # Octupole
        [(3, 2), (3, -2)],   # Mixed
        [(4, 4), (4, -4)],   # Hexadecapole
    ]

    hp_sum = np.zeros_like(hp_full)
    hc_sum = np.zeros_like(hc_full)

    mode_contributions = {}

    for mode_group in mode_groups:
        group_name = f"({mode_group[0][0]}, ±{abs(mode_group[0][1])})"
        print(f"   Generating modes {mode_group}...")

        _, hp_mode, hc_mode = generate_fd_waveform(params, mode_array=mode_group)

        hp_sum += hp_mode
        hc_sum += hc_mode

        mode_contributions[group_name] = {
            'hp': hp_mode.copy(),
            'hc': hc_mode.copy(),
        }

    # 3. Compare sum vs full
    print("\n3. Comparing sum of modes vs full waveform...")

    # Find valid frequency range
    mask = (freqs >= params.f_min) & (freqs <= params.f_max)

    diff_hp = np.abs(hp_full[mask] - hp_sum[mask])
    diff_hc = np.abs(hc_full[mask] - hc_sum[mask])

    amp_full = np.abs(hp_full[mask])
    rel_diff_hp = diff_hp / (amp_full + 1e-50)

    print(f"\n--- Results ---")
    print(f"Max |h_full - sum(h_lm)| for h+:  {np.max(diff_hp):.3e}")
    print(f"Max |h_full - sum(h_lm)| for hx:  {np.max(diff_hc):.3e}")
    print(f"Max relative difference for h+:   {np.max(rel_diff_hp):.3e}")

    # Check if additive (use relative tolerance based on machine precision)
    # Machine precision is ~1e-15 for float64, accumulated errors ~1e-9 is excellent
    rel_tolerance = 1e-8
    max_rel_error = np.max(diff_hp) / np.max(amp_full)
    is_additive = max_rel_error < rel_tolerance

    print(f"\n{'✓' if is_additive else '✗'} Modes ARE {'additive' if is_additive else 'NOT additive'}")
    print(f"  Max relative error: {max_rel_error:.3e}")
    print(f"  (tolerance: {rel_tolerance:.3e})")

    # Compute match
    def compute_match(h1, h2, freqs, f_low=20.0):
        mask = freqs >= f_low
        h1_m, h2_m = h1[mask], h2[mask]
        df = freqs[1] - freqs[0]
        inner_11 = 4 * df * np.sum(np.abs(h1_m)**2).real
        inner_22 = 4 * df * np.sum(np.abs(h2_m)**2).real
        inner_12 = 4 * df * np.sum(h1_m * np.conj(h2_m))
        return np.abs(inner_12) / np.sqrt(inner_11 * inner_22)

    match_hp = compute_match(hp_full, hp_sum, freqs)
    match_hc = compute_match(hc_full, hc_sum, freqs)

    print(f"\nMatch(h+_full, sum(h+_lm)): {match_hp:.15f}")
    print(f"Match(hx_full, sum(hx_lm)): {match_hc:.15f}")
    print(f"Mismatch h+: {1 - match_hp:.3e}")
    print(f"Mismatch hx: {1 - match_hc:.3e}")

    # 4. Plot comparison
    print("\n4. Generating comparison plots...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Mode Additivity Test: h(f) vs sum of h_lm(f)", fontsize=14)

    freqs_plot = freqs[mask]

    # Top row: Amplitude comparison
    ax = axes[0, 0]
    ax.semilogy(freqs_plot, np.abs(hp_full[mask]), 'b-', label='Full waveform', linewidth=2)
    ax.semilogy(freqs_plot, np.abs(hp_sum[mask]), 'r--', label='Sum of modes', linewidth=1.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|h+(f)|")
    ax.set_title("Amplitude: Full vs Sum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Individual mode contributions
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(mode_contributions)))
    for (name, data), color in zip(mode_contributions.items(), colors):
        ax.semilogy(freqs_plot, np.abs(data['hp'][mask]) + 1e-50,
                   label=name, color=color, linewidth=1.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|h+(f)|")
    ax.set_title("Individual Mode Contributions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle row: Phase comparison
    ax = axes[1, 0]
    phase_full = np.unwrap(np.angle(hp_full[mask]))
    phase_sum = np.unwrap(np.angle(hp_sum[mask]))
    ax.plot(freqs_plot, phase_full, 'b-', label='Full', linewidth=2)
    ax.plot(freqs_plot, phase_sum, 'r--', label='Sum', linewidth=1.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Phase [rad]")
    ax.set_title("Phase: Full vs Sum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phase difference
    ax = axes[1, 1]
    phase_diff = phase_full - phase_sum
    ax.plot(freqs_plot, phase_diff, 'g-', linewidth=1.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Phase difference [rad]")
    ax.set_title("Phase(full) - Phase(sum)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)

    # Bottom row: Residuals
    ax = axes[2, 0]
    ax.semilogy(freqs_plot, diff_hp, 'b-', linewidth=1.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|h+_full - h+_sum|")
    ax.set_title("Absolute Difference")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.semilogy(freqs_plot, rel_diff_hp + 1e-16, 'r-', linewidth=1.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Relative difference")
    ax.set_title("Relative Difference |diff|/|h_full|")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/mode_additivity_test.png", dpi=150)
    print(f"\nPlot saved to: figures/mode_additivity_test.png")
    plt.close()

    return is_additive


def test_individual_modes_separately():
    """Test generating each individual mode separately (not grouped)."""

    params = WaveformParameters(
        mass_1=35.0,
        mass_2=25.0,
        chi1z=0.3,
        chi2z=-0.2,
        luminosity_distance=100.0,
        inclination=np.pi / 4,
        f_min=20.0,
        f_max=512.0,
        delta_f=0.125,
        approximant="IMRPhenomXPHM",
    )

    print("\n" + "=" * 70)
    print("Testing Individual Mode Generation")
    print("=" * 70)

    # Generate full waveform
    freqs, hp_full, _ = generate_fd_waveform(params, mode_array=None)

    # Generate each mode individually
    individual_modes = [(2, 2), (2, -2), (2, 1), (2, -1), (3, 3), (3, -3),
                        (3, 2), (3, -2), (4, 4), (4, -4)]

    hp_sum_individual = np.zeros_like(hp_full)

    print("\nGenerating individual modes:")
    for mode in individual_modes:
        print(f"  Mode {mode}...", end=" ")
        _, hp_mode, _ = generate_fd_waveform(params, mode_array=[mode])
        hp_sum_individual += hp_mode
        print(f"max|h| = {np.max(np.abs(hp_mode)):.3e}")

    # Compare
    mask = (freqs >= params.f_min) & (freqs <= params.f_max)
    diff = np.abs(hp_full[mask] - hp_sum_individual[mask])

    print(f"\nMax |h_full - sum(individual modes)|: {np.max(diff):.3e}")

    max_rel_error = np.max(diff) / np.max(np.abs(hp_full[mask]))
    rel_tolerance = 1e-8  # Numerical precision tolerance
    is_additive = max_rel_error < rel_tolerance
    print(f"Max relative error: {max_rel_error:.3e}")
    print(f"{'✓' if is_additive else '✗'} Individual modes ARE {'additive' if is_additive else 'NOT additive'}")

    # Important finding about LAL convention
    print("\n--- Important Finding ---")
    print("Negative m modes (2,-2), (3,-3), etc. give ZERO when generated alone.")
    print("This means LAL includes BOTH +m and -m contributions when you request (l, +m).")
    print("For mode selection, use only positive m: [(2,2), (2,1), (3,3), (3,2), (4,4)]")


if __name__ == "__main__":
    # Test grouped modes
    test_mode_additivity()

    # Test individual modes
    test_individual_modes_separately()
