#!/usr/bin/env python3
"""
Deep investigation of individual mode phase convention.

Key questions:
1. What is the phase at f_ref when phi_c=0?
2. How does phi_c affect the phase?
3. Are phase differences between frequencies preserved?
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
D_L = 100.0
F_REF = 50.0

MSUN_SI = lal.MSUN_SI
MPC_SI = lal.PC_SI * 1e6


def get_component_masses(m_total, q):
    m1 = m_total * q / (1 + q)
    m2 = m_total / (1 + q)
    return m1, m2


def generate_mode(frequencies, phi_c, f_ref):
    """Generate (2,2) mode at specified frequencies."""
    m1, m2 = get_component_masses(M_TOTAL, Q)
    return generate_fd_mode_at_frequencies(
        frequencies=np.array(frequencies),
        mass_1=m1,
        mass_2=m2,
        chi1z=CHI1,
        chi2z=CHI2,
        ell=2,
        emm=2,
        luminosity_distance=D_L,
        phase=phi_c,
        f_ref=f_ref,
    )


def wrap_phase(p):
    return (p + np.pi) % (2*np.pi) - np.pi


def main():
    print("=" * 70)
    print(" INDIVIDUAL MODE PHASE CONVENTION")
    print("=" * 70)

    # Test 1: Phase at f_ref for different phi_c
    print("\n--- Test 1: Wrapped phase at f_ref for different phi_c ---")
    print(f"{'phi_c':>10} | {'angle(h)':>12} | {'Expected -π/4+?':>15}")
    print("-" * 45)

    for phi_c in [0.0, np.pi/4, np.pi/2, np.pi]:
        h = generate_mode([F_REF], phi_c, F_REF)
        phase = np.angle(h[0])
        # Try different formulas
        exp1 = wrap_phase(-np.pi/4 - 2*phi_c)  # -π/4 - 2φ_c
        exp2 = wrap_phase(-np.pi/4 + 2*phi_c)  # -π/4 + 2φ_c
        print(f"{phi_c:>10.4f} | {phase:>12.6f} | -π/4 - 2φ = {exp1:.4f}")

    # Test 2: Check if it's -π/4 - 2*phi_c
    print("\n--- Test 2: Verify formula angle(h) = -π/4 - 2*phi_c ---")
    print(f"{'phi_c':>10} | {'Predicted':>12} | {'Actual':>12} | {'Match':>6}")
    print("-" * 50)

    all_match = True
    for phi_c in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
        h = generate_mode([F_REF], phi_c, F_REF)
        actual = np.angle(h[0])
        predicted = wrap_phase(-np.pi/4 - 2*phi_c)
        diff = abs(wrap_phase(actual - predicted))
        match = diff < 0.001
        all_match = all_match and match
        print(f"{phi_c:>10.4f} | {predicted:>12.6f} | {actual:>12.6f} | {'✓' if match else '✗'}")

    print(f"\nFormula verified: {'YES' if all_match else 'NO'}")

    # Test 3: Phase differences between frequencies (should be f_ref independent)
    print("\n--- Test 3: Phase differences are f_ref independent ---")

    f1, f2 = 40.0, 100.0

    for f_ref_test in [30.0, 50.0, 100.0]:
        h = generate_mode([f1, f2], 0.0, f_ref_test)
        phase = np.unwrap(np.angle(h))
        diff = phase[1] - phase[0]
        print(f"f_ref = {f_ref_test:>5.0f} Hz: Φ({f2}) - Φ({f1}) = {diff:.6f}")

    # Test 4: What the training stores and inference needs
    print("\n" + "=" * 70)
    print(" IMPLICATIONS FOR TRAINING/INFERENCE")
    print("=" * 70)

    print("""
For INDIVIDUAL MODES (SimIMRPhenomXHMFrequencySequenceOneMode):

The convention is:
    angle(h_lm at f_ref) = -π/4 - 2*phi_c

    (Note: MINUS 2*phi_c, not plus!)

TRAINING (what generate_data.py does):
    - Call with phase=0.0, f_ref=f_ref_train
    - phase_raw = np.unwrap(np.angle(h_lm))
    - At f_ref: phase_raw = -π/4 (since phi_c=0)
    - Store: phase_stored = phase_raw - phase_raw[f_ref_train]

    Result: phase_stored(f_ref_train) = 0

INFERENCE (to match LAL with arbitrary phi_c, f_ref):
    LAL gives: phase_LAL = phase_intrinsic - π/4 - 2*phi_c

    Since phase_stored = phase_intrinsic - phase_intrinsic(f_ref_train) - (-π/4)
                       = phase_intrinsic - phase_intrinsic(f_ref_train) + π/4

    To reconstruct:
    phase_LAL(f) = phase_stored(f) - phase_stored(f_ref_inf) - 2*phi_c - π/4

    Note the MINUS sign on phi_c!
""")

    # Test 5: Verify the inference formula
    print("\n--- Test 5: Verify inference formula ---")

    f_ref_train = 50.0
    f_ref_inf = 30.0
    phi_c_test = np.pi / 3

    # Generate "training" data (phi_c=0)
    freqs = np.array([30.0, 50.0, 100.0, 200.0])
    h_train = generate_mode(freqs, 0.0, f_ref_train)
    phase_train = np.unwrap(np.angle(h_train))

    # What we store
    idx_ref_train = 1  # f=50Hz
    phase_stored = phase_train - phase_train[idx_ref_train]

    # Inference: reconstruct with different f_ref and phi_c
    idx_ref_inf = 0  # f=30Hz
    phase_reconstructed = phase_stored - phase_stored[idx_ref_inf] - 2*phi_c_test - np.pi/4

    # Ground truth: generate directly with target parameters
    h_truth = generate_mode(freqs, phi_c_test, f_ref_inf)
    phase_truth = np.unwrap(np.angle(h_truth))

    print(f"Training: f_ref = {f_ref_train} Hz, phi_c = 0")
    print(f"Inference: f_ref = {f_ref_inf} Hz, phi_c = {phi_c_test:.4f}")
    print()
    print(f"{'Freq':>8} | {'Reconstructed':>14} | {'Truth':>14} | {'Diff':>10}")
    print("-" * 55)

    all_pass = True
    for i, f in enumerate(freqs):
        diff = wrap_phase(phase_reconstructed[i] - phase_truth[i])
        match = abs(diff) < 0.01
        all_pass = all_pass and match
        print(f"{f:>8.0f} | {phase_reconstructed[i]:>14.6f} | {phase_truth[i]:>14.6f} | {diff:>9.6f} {'✓' if match else '✗'}")

    print("-" * 55)
    print(f"Inference formula verified: {'YES' if all_pass else 'NO'}")

    # Final summary
    print("\n" + "=" * 70)
    print(" FINAL ANSWER")
    print("=" * 70)
    print("""
For individual modes via generate_fd_mode_at_frequencies():

    PHASE OFFSET = -π/4   (not -3π/4!)
    PHI_C SIGN = NEGATIVE (not positive!)

The inference formula is:

    phase = phase_stored(f) - phase_stored(f_ref_inf) - 2*phi_c - π/4
                                                       ^^^^^^^^^
                                                       MINUS, not plus!

This differs from the full waveform convention!
""")


if __name__ == "__main__":
    main()
