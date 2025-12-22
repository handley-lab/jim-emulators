#!/usr/bin/env python3
"""
Verify that individual mode extraction has the same phase convention
as the full waveform.

The question: Does SimIMRPhenomXHMFrequencySequenceOneMode (used in generate_data.py)
have the same -3π/4 offset as SimIMRPhenomXHM (tested in learn_phase_conventions_v2.py)?
"""

import numpy as np
import lalsimulation as lalsim
import lal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.waveforms import generate_fd_mode_at_frequencies

# Parameters
M_TOTAL = 30.0
Q = 1.5
CHI1 = 0.2
CHI2 = 0.1
D_L = 100.0  # Mpc
F_REF = 50.0  # Hz

MSUN_SI = lal.MSUN_SI
MPC_SI = lal.PC_SI * 1e6


def get_component_masses(m_total, q):
    m1 = m_total * q / (1 + q)
    m2 = m_total / (1 + q)
    return m1, m2


def test_full_waveform_phase():
    """Test phase from SimIMRPhenomXHM (full waveform)."""
    m1, m2 = get_component_masses(M_TOTAL, Q)

    hp, hc = lalsim.SimIMRPhenomXHM(
        m1 * MSUN_SI, m2 * MSUN_SI,
        CHI1, CHI2,
        20.0, 1024.0, 0.125,  # f_min, f_max, delta_f
        D_L * MPC_SI,
        0.0,  # inclination (face-on)
        0.0,  # phi_c = 0
        F_REF,
        lal.CreateDict()
    )

    idx_ref = int(F_REF / 0.125)
    return np.angle(hp.data.data[idx_ref])


def test_individual_mode_phase():
    """Test phase from SimIMRPhenomXHMFrequencySequenceOneMode (individual mode)."""
    m1, m2 = get_component_masses(M_TOTAL, Q)

    # Use the function from generate_data.py
    freqs = np.array([F_REF])
    h22 = generate_fd_mode_at_frequencies(
        frequencies=freqs,
        mass_1=m1,
        mass_2=m2,
        chi1z=CHI1,
        chi2z=CHI2,
        ell=2,
        emm=2,
        luminosity_distance=D_L,
        phase=0.0,
        f_ref=F_REF,
    )

    return np.angle(h22[0])


def test_phi_c_effect_on_mode():
    """Verify that phi_c affects mode phase the same way as full waveform."""
    m1, m2 = get_component_masses(M_TOTAL, Q)
    freqs = np.array([F_REF])

    phases = []
    for phi_c in [0.0, np.pi/4, np.pi/2]:
        h22 = generate_fd_mode_at_frequencies(
            frequencies=freqs,
            mass_1=m1,
            mass_2=m2,
            chi1z=CHI1,
            chi2z=CHI2,
            ell=2,
            emm=2,
            luminosity_distance=D_L,
            phase=phi_c,
            f_ref=F_REF,
        )
        phases.append(np.angle(h22[0]))

    return phases


def main():
    print("=" * 70)
    print(" VERIFYING MODE PHASE CONVENTION")
    print("=" * 70)

    # Test 1: Compare full waveform vs individual mode at f_ref
    print("\n--- Test 1: Phase at f_ref with phi_c=0 ---")
    phase_full = test_full_waveform_phase()
    phase_mode = test_individual_mode_phase()

    print(f"Full waveform (SimIMRPhenomXHM):     {phase_full:.6f} rad")
    print(f"Individual mode (FrequencySequence): {phase_mode:.6f} rad")
    print(f"Difference: {phase_full - phase_mode:.6f} rad")

    expected = -3 * np.pi / 4
    print(f"\nExpected offset (-3π/4): {expected:.6f} rad")
    print(f"Full waveform matches: {abs(phase_full - expected) < 0.01}")
    print(f"Individual mode matches: {abs(phase_mode - expected) < 0.01}")

    # Test 2: Verify phi_c effect on individual mode
    print("\n--- Test 2: phi_c effect on individual mode ---")
    phases = test_phi_c_effect_on_mode()

    print(f"phi_c = 0:    phase = {phases[0]:.6f}")
    print(f"phi_c = π/4:  phase = {phases[1]:.6f}")
    print(f"phi_c = π/2:  phase = {phases[2]:.6f}")

    diff1 = phases[1] - phases[0]
    diff2 = phases[2] - phases[1]

    print(f"\nPhase shift for Δphi_c = π/4:")
    print(f"  Expected: 2×(π/4) = {np.pi/2:.6f}")
    print(f"  Actual (0→π/4):   {diff1:.6f}")
    print(f"  Actual (π/4→π/2): {diff2:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    mode_offset_correct = abs(phase_mode - expected) < 0.01
    phi_c_effect_correct = abs(diff1 - np.pi/2) < 0.01 and abs(diff2 - np.pi/2) < 0.01

    if mode_offset_correct and phi_c_effect_correct:
        print("""
✓ Individual mode extraction has the SAME phase convention as full waveform!

The current generate_data.py implementation is CORRECT:
- Uses phase=0.0 and f_ref correctly
- Phase at f_ref with phi_c=0 is -3π/4
- Changing phi_c shifts phase by 2×Δphi_c

The inference formula should be:
    phase = phase_stored(f) - phase_stored(f_ref_inf) + 2*phi_c - 3π/4
""")
    else:
        print("✗ Mismatch detected! Need to investigate.")
        if not mode_offset_correct:
            print(f"  - Mode offset: expected {expected:.4f}, got {phase_mode:.4f}")
        if not phi_c_effect_correct:
            print(f"  - phi_c effect: expected {np.pi/2:.4f}, got {diff1:.4f}")


if __name__ == "__main__":
    main()
