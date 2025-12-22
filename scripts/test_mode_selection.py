#!/usr/bin/env python
"""
Test that IMRPhenomXPHM with only (2,2) mode gives the same result as IMRPhenomXAS.

This verifies that the mode selection mechanism works correctly.
"""

import numpy as np
import lal
import lalsimulation as lalsim
import matplotlib.pyplot as plt


def generate_waveform_with_modes(
    m1_msun: float,
    m2_msun: float,
    chi1z: float,
    chi2z: float,
    distance_mpc: float,
    inclination: float,
    f_min: float,
    f_max: float,
    delta_f: float,
    approximant: str,
    mode_array: list = None,
):
    """
    Generate a frequency-domain waveform with optional mode selection.

    Parameters
    ----------
    mode_array : list of tuples, optional
        List of (l, m) modes to include. If None, uses default (all modes).
        Example: [(2, 2), (2, -2)] for only the dominant quadrupole.
    """
    # Convert to SI units
    m1_kg = m1_msun * lal.MSUN_SI
    m2_kg = m2_msun * lal.MSUN_SI
    distance = distance_mpc * 1e6 * lal.PC_SI

    # Create LAL dictionary for extra parameters
    laldict = lal.CreateDict()

    # Set up mode array if specified
    if mode_array is not None:
        lal_mode_array = lalsim.SimInspiralCreateModeArray()
        for l, m in mode_array:
            lalsim.SimInspiralModeArrayActivateMode(lal_mode_array, l, m)
        lalsim.SimInspiralWaveformParamsInsertModeArray(laldict, lal_mode_array)

    # Get approximant
    approx = lalsim.GetApproximantFromString(approximant)

    # Generate waveform
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_kg,                    # mass1
        m2_kg,                    # mass2
        0.0, 0.0, chi1z,         # spin1 (x, y, z)
        0.0, 0.0, chi2z,         # spin2 (x, y, z)
        distance,                 # distance
        inclination,              # inclination
        0.0,                      # phiRef
        0.0,                      # longAscNodes
        0.0,                      # eccentricity
        0.0,                      # meanPerAno
        delta_f,                  # deltaF
        f_min,                    # f_min
        f_max,                    # f_max
        0.0,                      # f_ref (0 = use f_min)
        laldict,                  # LALDict
        approx,                   # approximant
    )

    # Extract data
    n_freqs = hp.data.length
    freqs = np.arange(n_freqs) * delta_f
    hp_data = hp.data.data
    hc_data = hc.data.data

    return freqs, hp_data, hc_data


def main():
    # Test parameters - use aligned spins for fair comparison
    params = {
        "m1_msun": 35.0,
        "m2_msun": 25.0,
        "chi1z": 0.3,
        "chi2z": -0.2,
        "distance_mpc": 100.0,
        "inclination": np.pi / 4,
        "f_min": 20.0,
        "f_max": 512.0,
        "delta_f": 0.125,
    }

    print("=" * 60)
    print("Testing: XPHM with (2,±2) modes only vs XAS")
    print("=" * 60)
    print(f"\nParameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Generate XAS waveform (dominant mode only by design)
    print("\n1. Generating IMRPhenomXAS waveform...")
    freqs_xas, hp_xas, hc_xas = generate_waveform_with_modes(
        **params,
        approximant="IMRPhenomXAS",
        mode_array=None,  # XAS only has (2,2) anyway
    )

    # Generate XPHM waveform with all modes (default)
    print("2. Generating IMRPhenomXPHM waveform (all modes)...")
    freqs_xphm_all, hp_xphm_all, hc_xphm_all = generate_waveform_with_modes(
        **params,
        approximant="IMRPhenomXPHM",
        mode_array=None,
    )

    # Generate XPHM waveform with only (2,2) and (2,-2) modes
    print("3. Generating IMRPhenomXPHM waveform with only (2,±2) modes...")
    freqs_xphm_22, hp_xphm_22, hc_xphm_22 = generate_waveform_with_modes(
        **params,
        approximant="IMRPhenomXPHM",
        mode_array=[(2, 2), (2, -2)],
    )

    # Compare
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    # Find valid frequency range
    mask = (freqs_xas >= params["f_min"]) & (freqs_xas <= params["f_max"])

    # Compute differences
    diff_xas_vs_xphm_all = np.abs(hp_xas[mask] - hp_xphm_all[mask])
    diff_xas_vs_xphm_22 = np.abs(hp_xas[mask] - hp_xphm_22[mask])
    diff_xphm_22_vs_xphm_all = np.abs(hp_xphm_22[mask] - hp_xphm_all[mask])

    # Normalize by amplitude
    amp_xas = np.abs(hp_xas[mask])
    amp_xas_max = np.max(amp_xas)

    rel_diff_xas_vs_xphm_all = diff_xas_vs_xphm_all / (amp_xas + 1e-50)
    rel_diff_xas_vs_xphm_22 = diff_xas_vs_xphm_22 / (amp_xas + 1e-50)

    print(f"\nMax |XAS - XPHM(all modes)|:     {np.max(diff_xas_vs_xphm_all):.3e}")
    print(f"Max |XAS - XPHM(2,2 only)|:      {np.max(diff_xas_vs_xphm_22):.3e}")
    print(f"Max |XPHM(2,2) - XPHM(all)|:     {np.max(diff_xphm_22_vs_xphm_all):.3e}")

    print(f"\nMax relative |XAS - XPHM(all)|:  {np.max(rel_diff_xas_vs_xphm_all):.3e}")
    print(f"Max relative |XAS - XPHM(2,2)|:  {np.max(rel_diff_xas_vs_xphm_22):.3e}")

    # Compute match (overlap) - the proper GW metric
    # Using flat PSD for simplicity
    def compute_match(h1, h2, freqs, f_low=20.0):
        """Compute match between two waveforms (max overlap over phase)."""
        mask = freqs >= f_low
        h1_m = h1[mask]
        h2_m = h2[mask]
        df = freqs[1] - freqs[0]

        # Inner products (assuming flat PSD)
        inner_11 = 4 * df * np.sum(np.abs(h1_m)**2).real
        inner_22 = 4 * df * np.sum(np.abs(h2_m)**2).real
        inner_12 = 4 * df * np.sum(h1_m * np.conj(h2_m))

        # Match = |<h1|h2>| / sqrt(<h1|h1> * <h2|h2>)
        return np.abs(inner_12) / np.sqrt(inner_11 * inner_22)

    match_xas_xphm_all = compute_match(hp_xas, hp_xphm_all, freqs_xas)
    match_xas_xphm_22 = compute_match(hp_xas, hp_xphm_22, freqs_xas)
    match_xphm_22_xphm_all = compute_match(hp_xphm_22, hp_xphm_all, freqs_xas)

    print(f"\n--- Match (overlap) ---")
    print(f"Match(XAS, XPHM all modes):      {match_xas_xphm_all:.10f}")
    print(f"Match(XAS, XPHM 2,±2 only):      {match_xas_xphm_22:.10f}")
    print(f"Match(XPHM 2,2, XPHM all):       {match_xphm_22_xphm_all:.10f}")

    print(f"\n--- Mismatch (1 - match) ---")
    print(f"Mismatch(XAS, XPHM all):         {1 - match_xas_xphm_all:.6e}")
    print(f"Mismatch(XAS, XPHM 2,±2):        {1 - match_xas_xphm_22:.6e}")
    print(f"Mismatch(XPHM 2,2, XPHM all):    {1 - match_xphm_22_xphm_all:.6e}")

    # Interpretation
    print(f"\n--- Interpretation ---")
    if match_xas_xphm_22 > 0.999:
        print(f"✓ XPHM(2,±2) is effectively equivalent to XAS (match > 0.999)")
    else:
        print(f"✗ XPHM(2,±2) differs from XAS (match = {match_xas_xphm_22:.6f})")

    print(f"\nHigher modes contribution: {1 - match_xphm_22_xphm_all:.4e} mismatch")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparison: XAS vs XPHM with mode selection", fontsize=14)

    freqs_plot = freqs_xas[mask]

    # Amplitude comparison
    ax = axes[0, 0]
    ax.semilogy(freqs_plot, np.abs(hp_xas[mask]), 'b-', label='XAS', linewidth=2)
    ax.semilogy(freqs_plot, np.abs(hp_xphm_all[mask]), 'r--', label='XPHM (all modes)', linewidth=1.5)
    ax.semilogy(freqs_plot, np.abs(hp_xphm_22[mask]), 'g:', label='XPHM (2,±2 only)', linewidth=2)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|h+(f)|")
    ax.set_title("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phase comparison
    ax = axes[0, 1]
    phase_xas = np.unwrap(np.angle(hp_xas[mask]))
    phase_xphm_all = np.unwrap(np.angle(hp_xphm_all[mask]))
    phase_xphm_22 = np.unwrap(np.angle(hp_xphm_22[mask]))
    ax.plot(freqs_plot, phase_xas, 'b-', label='XAS', linewidth=2)
    ax.plot(freqs_plot, phase_xphm_all, 'r--', label='XPHM (all modes)', linewidth=1.5)
    ax.plot(freqs_plot, phase_xphm_22, 'g:', label='XPHM (2,±2 only)', linewidth=2)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Phase [rad]")
    ax.set_title("Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Absolute difference
    ax = axes[1, 0]
    ax.semilogy(freqs_plot, diff_xas_vs_xphm_all, 'r-', label='|XAS - XPHM(all)|', linewidth=1.5)
    ax.semilogy(freqs_plot, diff_xas_vs_xphm_22, 'g-', label='|XAS - XPHM(2,2)|', linewidth=1.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Absolute difference")
    ax.set_title("Absolute Difference from XAS")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative difference
    ax = axes[1, 1]
    ax.semilogy(freqs_plot, rel_diff_xas_vs_xphm_all, 'r-', label='|XAS - XPHM(all)|/|XAS|', linewidth=1.5)
    ax.semilogy(freqs_plot, rel_diff_xas_vs_xphm_22 + 1e-16, 'g-', label='|XAS - XPHM(2,2)|/|XAS|', linewidth=1.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Relative difference")
    ax.set_title("Relative Difference from XAS")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/mode_selection_test.png", dpi=150)
    print(f"\nPlot saved to: figures/mode_selection_test.png")
    plt.close()


if __name__ == "__main__":
    main()
