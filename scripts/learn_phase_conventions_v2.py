#!/usr/bin/env python3
"""
Learn Phase Conventions in LAL - CORRECTED VERSION
===================================================

After investigation, we found the ACTUAL convention used by LAL:

    angle(h_plus at f_ref) = 2×phi_c - 3π/4

NOT the +π/4 from the docs! The -3π/4 = -135° comes from:
- The stationary phase approximation gives π/4
- But there's an additional -π from LAL's internal conventions

The key relationships for EMULATION:
1. Changing phi_c by Δ shifts phase by +2Δ at ALL frequencies
2. Phase differences between frequencies are INDEPENDENT of f_ref
3. Time shift tc adds 2π×f×tc to the phase

For training: Store phase relative to a reference frequency
For inference: Apply the offset correction with the corrected constant
"""

import numpy as np
import lalsimulation as lalsim
import lal

# Fixed parameters
M_TOTAL = 30.0
Q = 1.5
CHI1 = 0.2
CHI2 = 0.1
D_L = 100.0
INCLINATION = 0.0
F_MIN = 20.0
F_MAX = 1024.0
DELTA_F = 0.125

# THE CORRECTED CONSTANT!
PHASE_OFFSET = -3 * np.pi / 4  # NOT +π/4!


def get_component_masses(m_total, q):
    m1 = m_total * q / (1 + q)
    m2 = m_total / (1 + q)
    return m1, m2


def generate_waveform(f_ref, phi_c):
    m1, m2 = get_component_masses(M_TOTAL, Q)
    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    d_l_m = D_L * lal.PC_SI * 1e6

    hp, hc = lalsim.SimIMRPhenomXHM(
        m1_kg, m2_kg, CHI1, CHI2,
        F_MIN, F_MAX, DELTA_F,
        d_l_m, INCLINATION,
        phi_c, f_ref,
        lal.CreateDict()
    )

    h_plus = hp.data.data
    n_samples = len(h_plus)
    frequencies = np.arange(n_samples) * DELTA_F

    return frequencies, h_plus


def extract_phase(h, frequencies, f_min=F_MIN):
    mask = (frequencies >= f_min) & (np.abs(h) > 0)
    phase = np.angle(h)
    phase_unwrapped = np.unwrap(phase)
    return frequencies[mask], phase_unwrapped[mask]


def phase_at_frequency(freqs, phase, f_target):
    return np.interp(f_target, freqs, phase)


def wrap_phase(phase):
    """Wrap phase to [-π, π]."""
    return (phase + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# TEST 1: Phase at f_ref with CORRECTED formula
# =============================================================================
def test_phase_at_fref_corrected():
    """
    CORRECTED PREDICTION: At f = f_ref, the WRAPPED phase should be:
        angle(h) mod 2π = 2*phi_c - 3π/4

    The offset is -3π/4, NOT +π/4!
    """
    print("=" * 70)
    print("TEST 1: Phase at f_ref equals 2*phi_c - 3π/4 (CORRECTED)")
    print("=" * 70)

    f_ref = 50.0

    test_cases = [
        {"phi_c": 0.0},
        {"phi_c": np.pi/4},
        {"phi_c": np.pi/2},
        {"phi_c": np.pi},
    ]

    print(f"\nf_ref = {f_ref} Hz")
    print(f"Prediction: angle(h) at f_ref = 2*phi_c - 3π/4 (wrapped)")
    print("-" * 70)
    print(f"{'phi_c':>10} | {'Predicted':>12} | {'Actual':>12} | {'Diff':>10}")
    print("-" * 70)

    all_pass = True
    for case in test_cases:
        phi_c = case["phi_c"]
        expected = wrap_phase(2 * phi_c + PHASE_OFFSET)

        freqs, h = generate_waveform(f_ref=f_ref, phi_c=phi_c)
        idx = int(f_ref / DELTA_F)
        actual = np.angle(h[idx])

        diff = wrap_phase(actual - expected)
        status = "✓" if abs(diff) < 0.01 else "✗"
        all_pass = all_pass and (abs(diff) < 0.01)

        print(f"{phi_c:>10.4f} | {expected:>12.6f} | {actual:>12.6f} | {diff:>9.6f} {status}")

    print("-" * 70)
    print(f"TEST 1: {'PASSED' if all_pass else 'FAILED'}")
    print()
    return all_pass


# =============================================================================
# TEST 2: Training → Inference with CORRECTED formula
# =============================================================================
def test_training_inference_corrected():
    """
    CORRECTED WORKFLOW:

    Training: Store Φ_stored(f) = Φ(f; f_ref_train, phi_c=0) - Φ(f_ref_train)

    Inference: Φ(f; f_ref_inf, phi_c) = Φ_stored(f) - Φ_stored(f_ref_inf) + 2*phi_c - 3π/4

    The key change: use -3π/4 instead of +π/4
    """
    print("=" * 70)
    print("TEST 2: Training → Inference with CORRECTED Formula")
    print("=" * 70)

    f_ref_train = 50.0
    f_ref_inf = 30.0
    phi_c_inf = np.pi / 3

    print(f"\nTraining: f_ref = {f_ref_train} Hz, phi_c = 0")
    print(f"Inference: f_ref = {f_ref_inf} Hz, phi_c = {phi_c_inf:.4f}")
    print(f"Using CORRECTED offset: -3π/4 = {PHASE_OFFSET:.4f}")
    print("-" * 70)

    # TRAINING: Generate with phi_c=0
    freqs_train, h_train = generate_waveform(f_ref=f_ref_train, phi_c=0.0)
    f_train, phase_train = extract_phase(h_train, freqs_train)

    # What we store: phase - phase(f_ref_train)
    phase_at_fref_train = phase_at_frequency(f_train, phase_train, f_ref_train)
    phase_stored = phase_train - phase_at_fref_train

    print(f"Stored phase at f_ref_train: {phase_at_frequency(f_train, phase_stored, f_ref_train):.6f} (should be 0)")

    # INFERENCE: Reconstruct with CORRECTED formula
    phase_stored_at_fref_inf = phase_at_frequency(f_train, phase_stored, f_ref_inf)

    # CORRECTED: use -3π/4 instead of +π/4
    phase_reconstructed = phase_stored - phase_stored_at_fref_inf + 2*phi_c_inf + PHASE_OFFSET

    # GROUND TRUTH
    freqs_truth, h_truth = generate_waveform(f_ref=f_ref_inf, phi_c=phi_c_inf)
    f_truth, phase_truth = extract_phase(h_truth, freqs_truth)

    # Compare
    test_freqs = [30.0, 50.0, 100.0, 200.0]

    print(f"\n{'Frequency':>10} | {'Reconstructed':>14} | {'LAL Truth':>14} | {'Diff':>10}")
    print("-" * 70)

    all_pass = True
    for f in test_freqs:
        if f > f_train[-1]:
            continue

        recon = phase_at_frequency(f_train, phase_reconstructed, f)
        truth = phase_at_frequency(f_truth, phase_truth, f)

        diff = wrap_phase(recon - truth)
        status = "✓" if abs(diff) < 0.01 else "✗"
        all_pass = all_pass and (abs(diff) < 0.01)

        print(f"{f:>10.1f} | {recon:>14.6f} | {truth:>14.6f} | {diff:>9.6f} {status}")

    print("-" * 70)
    print(f"TEST 2: {'PASSED' if all_pass else 'FAILED'}")
    print()
    return all_pass


# =============================================================================
# TEST 3: Verify the complete emulation workflow
# =============================================================================
def test_complete_emulation_workflow():
    """
    Complete end-to-end test of the emulation workflow.

    This simulates what we'll do with the neural network:
    1. Train on waveforms with phi_c=0, fixed f_ref_train
    2. Store phase relative to f_ref_train
    3. At inference, reconstruct with arbitrary f_ref and phi_c
    """
    print("=" * 70)
    print("TEST 3: Complete Emulation Workflow")
    print("=" * 70)

    f_ref_train = 50.0

    # Simulate multiple inference scenarios
    test_cases = [
        {"f_ref": 30.0, "phi_c": 0.0, "name": "Different f_ref, phi_c=0"},
        {"f_ref": 50.0, "phi_c": np.pi/3, "name": "Same f_ref, phi_c≠0"},
        {"f_ref": 100.0, "phi_c": np.pi/6, "name": "Higher f_ref, arbitrary phi_c"},
        {"f_ref": 40.0, "phi_c": np.pi, "name": "f_ref=40, phi_c=π"},
    ]

    # TRAINING PHASE
    print(f"\n--- TRAINING PHASE ---")
    print(f"f_ref_train = {f_ref_train} Hz, phi_c = 0")

    freqs_train, h_train = generate_waveform(f_ref=f_ref_train, phi_c=0.0)
    f_train, phase_train = extract_phase(h_train, freqs_train)

    phase_at_fref_train = phase_at_frequency(f_train, phase_train, f_ref_train)
    phase_stored = phase_train - phase_at_fref_train  # This is what NN learns

    print(f"Stored {len(phase_stored)} phase values from {f_train[0]:.1f} to {f_train[-1]:.1f} Hz")

    # INFERENCE PHASE
    print(f"\n--- INFERENCE PHASE ---")
    print("-" * 70)

    all_pass = True
    for case in test_cases:
        f_ref_inf = case["f_ref"]
        phi_c_inf = case["phi_c"]
        name = case["name"]

        print(f"\nCase: {name}")
        print(f"  f_ref = {f_ref_inf} Hz, phi_c = {phi_c_inf:.4f}")

        # Reconstruction formula (CORRECTED)
        phase_stored_at_fref_inf = phase_at_frequency(f_train, phase_stored, f_ref_inf)
        phase_reconstructed = phase_stored - phase_stored_at_fref_inf + 2*phi_c_inf + PHASE_OFFSET

        # Ground truth
        freqs_truth, h_truth = generate_waveform(f_ref=f_ref_inf, phi_c=phi_c_inf)
        f_truth, phase_truth = extract_phase(h_truth, freqs_truth)

        # Check at a few frequencies
        check_freqs = [50.0, 100.0, 200.0]
        max_diff = 0

        for f in check_freqs:
            if f > f_train[-1] or f > f_truth[-1]:
                continue
            recon = phase_at_frequency(f_train, phase_reconstructed, f)
            truth = phase_at_frequency(f_truth, phase_truth, f)
            diff = abs(wrap_phase(recon - truth))
            max_diff = max(max_diff, diff)

        status = "✓" if max_diff < 0.01 else "✗"
        all_pass = all_pass and (max_diff < 0.01)
        print(f"  Max phase error: {max_diff:.6f} {status}")

    print("-" * 70)
    print(f"TEST 3: {'PASSED' if all_pass else 'FAILED'}")
    print()
    return all_pass


# =============================================================================
# SUMMARY: The Emulation Recipe
# =============================================================================
def print_emulation_recipe():
    print("=" * 70)
    print(" THE EMULATION RECIPE (VERIFIED)")
    print("=" * 70)
    print("""
TRAINING:
---------
1. Choose a fixed f_ref_train (e.g., 50 Hz)
2. Generate waveforms with phi_c = 0
3. Extract unwrapped phase: phase = np.unwrap(np.angle(h_plus))
4. Store: phase_stored = phase - phase(f_ref_train)

   → The NN learns phase_stored(f; θ) where θ = intrinsic params

INFERENCE:
----------
Given: f_ref_inference, phi_c, and optional tc

1. Evaluate NN to get phase_stored at all frequencies
2. Get phase at inference reference: phase_stored(f_ref_inference)
3. Reconstruct:

   phase = phase_stored(f) - phase_stored(f_ref_inf) + 2*phi_c - 3π/4

4. If tc ≠ 0, add time shift:

   phase_final = phase + 2π × f × tc

KEY INSIGHT:
------------
The constant is -3π/4, NOT +π/4!

This comes from LAL's internal conventions combining:
- Stationary phase approximation: contributes π/4
- Sign conventions and mode factors: contribute -π

The -3π/4 = -135° appears at f_ref when phi_c = 0.
""")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print(" LEARNING LAL PHASE CONVENTIONS - CORRECTED VERSION")
    print("=" * 70 + "\n")

    results = []
    results.append(("Phase at f_ref (corrected)", test_phase_at_fref_corrected()))
    results.append(("Training→Inference (corrected)", test_training_inference_corrected()))
    results.append(("Complete workflow", test_complete_emulation_workflow()))

    print_emulation_recipe()

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:.<45} {status}")

    all_passed = all(r[1] for r in results)
    print("-" * 70)
    print(f"  {'OVERALL':.<45} {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
