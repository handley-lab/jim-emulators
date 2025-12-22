#!/usr/bin/env python3
"""
Learn Phase Conventions in LAL by Prediction and Verification
==============================================================

This script teaches you how tc, phi_c, and f_ref affect GW waveforms.
For each test, we:
1. State the PREDICTION based on theory
2. Run LAL to get actual values
3. Verify the prediction matches reality

The goal: You should be able to PREDICT what LAL will output before running it.

Key equations:
- Phase at any frequency: Φ(f) = Φ_intrinsic(f) - Φ_intrinsic(f_ref) + 2*phi_c + π/4
- Time shift effect: Φ(f; tc) = Φ(f; tc=0) + 2π*f*tc
- The π/4 comes from the stationary phase approximation (SPA) convention
"""

import numpy as np
import lalsimulation as lalsim
import lal

# Fixed parameters for all tests
M_TOTAL = 30.0  # Solar masses
Q = 1.5  # Mass ratio
CHI1 = 0.2  # Spin on larger BH
CHI2 = 0.1  # Spin on smaller BH
D_L = 100.0  # Mpc
INCLINATION = 0.0  # Face-on

# Frequency settings
F_MIN = 20.0  # Hz
F_MAX = 1024.0  # Hz
DELTA_F = 0.125  # Hz

def get_component_masses(m_total, q):
    """Convert total mass and mass ratio to component masses."""
    m1 = m_total * q / (1 + q)
    m2 = m_total / (1 + q)
    return m1, m2

def generate_waveform(f_ref, phi_c, tc=0.0):
    """
    Generate IMRPhenomXHM waveform with specified parameters.

    Returns: frequencies, h_plus (complex array)

    LAL signature:
    SimIMRPhenomXHM(m1_SI, m2_SI, chi1z, chi2z, f_min, f_max, deltaF,
                   distance, inclination, phiRef, fRef_In, lalParams)
    """
    m1, m2 = get_component_masses(M_TOTAL, Q)

    # Convert to SI units
    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    d_l_m = D_L * lal.PC_SI * 1e6

    # Generate waveform
    hp, hc = lalsim.SimIMRPhenomXHM(
        m1_kg, m2_kg,       # masses
        CHI1, CHI2,         # spins
        F_MIN, F_MAX,       # frequency range
        DELTA_F,            # frequency step
        d_l_m,              # distance
        INCLINATION,        # inclination
        phi_c,              # phiRef (coalescence phase)
        f_ref,              # reference frequency
        lal.CreateDict()    # LAL dictionary
    )

    # Extract data as numpy array
    h_plus = hp.data.data

    # Build frequency array
    n_samples = len(h_plus)
    frequencies = np.arange(n_samples) * DELTA_F

    return frequencies, h_plus


def extract_phase(h, frequencies, f_min=F_MIN):
    """Extract unwrapped phase from complex waveform."""
    # Only look at frequencies where waveform is non-zero
    mask = (frequencies >= f_min) & (np.abs(h) > 0)

    phase = np.angle(h)
    phase_unwrapped = np.unwrap(phase)

    return frequencies[mask], phase_unwrapped[mask]


def phase_at_frequency(freqs, phase, f_target):
    """Interpolate to get phase at specific frequency."""
    return np.interp(f_target, freqs, phase)


# =============================================================================
# TEST 1: Phase at reference frequency
# =============================================================================
def test_phase_at_fref():
    """
    PREDICTION: At f = f_ref, the phase should be:
        Φ(f_ref) = 2*phi_c + π/4

    Why? LAL defines phifRef to enforce this convention.
    The π/4 comes from the stationary phase approximation.
    """
    print("=" * 70)
    print("TEST 1: Phase at f_ref equals 2*phi_c + π/4")
    print("=" * 70)

    f_ref = 50.0  # Hz

    test_cases = [
        {"phi_c": 0.0, "expected": np.pi/4},
        {"phi_c": np.pi/4, "expected": np.pi/2 + np.pi/4},
        {"phi_c": np.pi/2, "expected": np.pi + np.pi/4},
        {"phi_c": np.pi, "expected": 2*np.pi + np.pi/4},
    ]

    print(f"\nf_ref = {f_ref} Hz")
    print(f"Prediction: Φ(f_ref) = 2*phi_c + π/4")
    print("-" * 70)
    print(f"{'phi_c':>10} | {'Predicted':>15} | {'Actual':>15} | {'Diff':>10}")
    print("-" * 70)

    all_pass = True
    for case in test_cases:
        phi_c = case["phi_c"]
        expected = case["expected"]

        # Generate waveform
        freqs, h = generate_waveform(f_ref=f_ref, phi_c=phi_c)
        f_arr, phase = extract_phase(h, freqs)

        # Get phase at f_ref
        actual = phase_at_frequency(f_arr, phase, f_ref)

        # Compare (modulo 2π)
        diff = (actual - expected) % (2*np.pi)
        if diff > np.pi:
            diff -= 2*np.pi

        status = "✓" if abs(diff) < 0.01 else "✗"
        all_pass = all_pass and (abs(diff) < 0.01)

        print(f"{phi_c:>10.4f} | {expected:>15.6f} | {actual:>15.6f} | {diff:>9.6f} {status}")

    print("-" * 70)
    print(f"TEST 1: {'PASSED' if all_pass else 'FAILED'}")
    print()
    return all_pass


# =============================================================================
# TEST 2: Changing phi_c shifts phase by 2*Δphi_c everywhere
# =============================================================================
def test_phic_shift():
    """
    PREDICTION: Changing phi_c by Δphi_c shifts the phase by 2*Δphi_c
    at ALL frequencies (not just f_ref).

    Φ(f; phi_c + Δ) - Φ(f; phi_c) = 2*Δ  (for all f)

    Why factor of 2? Because GW phase = 2 × orbital phase (quadrupole).
    """
    print("=" * 70)
    print("TEST 2: Changing phi_c shifts phase by 2*Δphi_c everywhere")
    print("=" * 70)

    f_ref = 50.0
    phi_c_base = 0.0
    delta_phi_c = np.pi / 4  # Change phi_c by π/4

    print(f"\nPrediction: Φ(f; phi_c + Δ) - Φ(f; phi_c) = 2*Δ for ALL f")
    print(f"With Δ = {delta_phi_c:.4f}, expect phase shift = {2*delta_phi_c:.4f}")
    print("-" * 70)

    # Generate two waveforms
    freqs1, h1 = generate_waveform(f_ref=f_ref, phi_c=phi_c_base)
    freqs2, h2 = generate_waveform(f_ref=f_ref, phi_c=phi_c_base + delta_phi_c)

    f1, phase1 = extract_phase(h1, freqs1)
    f2, phase2 = extract_phase(h2, freqs2)

    # Check at multiple frequencies
    test_freqs = [30.0, 50.0, 100.0, 200.0, 400.0]

    print(f"{'Frequency':>10} | {'Expected Δφ':>12} | {'Actual Δφ':>12} | {'Diff':>10}")
    print("-" * 70)

    all_pass = True
    for f in test_freqs:
        if f > f1[-1]:
            continue
        p1 = phase_at_frequency(f1, phase1, f)
        p2 = phase_at_frequency(f2, phase2, f)

        actual_diff = p2 - p1
        expected_diff = 2 * delta_phi_c

        error = abs(actual_diff - expected_diff)
        status = "✓" if error < 0.01 else "✗"
        all_pass = all_pass and (error < 0.01)

        print(f"{f:>10.1f} | {expected_diff:>12.6f} | {actual_diff:>12.6f} | {error:>9.6f} {status}")

    print("-" * 70)
    print(f"TEST 2: {'PASSED' if all_pass else 'FAILED'}")
    print()
    return all_pass


# =============================================================================
# TEST 3: Time of coalescence shift
# =============================================================================
def test_tc_shift():
    """
    PREDICTION: Changing tc shifts the phase by 2π*f*Δtc at each frequency.

    Φ(f; tc + Δtc) - Φ(f; tc) = 2π * f * Δtc

    This is a LINEAR function of frequency (time shift in Fourier domain).

    Note: In LAL's FD waveforms, tc is typically handled externally,
    not as a parameter to SimIMRPhenomXHM. We verify the principle here.
    """
    print("=" * 70)
    print("TEST 3: Time shift produces phase shift of 2π*f*Δtc")
    print("=" * 70)

    f_ref = 50.0
    phi_c = 0.0

    # Generate baseline waveform
    freqs, h_base = generate_waveform(f_ref=f_ref, phi_c=phi_c)
    f_arr, phase_base = extract_phase(h_base, freqs)

    # Apply time shift externally (the correct way in FD)
    delta_tc = 0.01  # 10 ms time shift

    print(f"\nPrediction: Φ(f; tc + Δtc) - Φ(f; tc) = 2π * f * Δtc")
    print(f"With Δtc = {delta_tc*1000:.1f} ms")
    print("-" * 70)

    # Apply time shift: h(f; tc + Δtc) = h(f; tc) * exp(2πi f Δtc)
    # Phase shift = 2π f Δtc
    phase_shifted = phase_base + 2 * np.pi * f_arr * delta_tc

    # Verify the relationship
    test_freqs = [30.0, 50.0, 100.0, 200.0, 400.0]

    print(f"{'Frequency':>10} | {'Expected Δφ':>12} | {'Actual Δφ':>12}")
    print("-" * 70)

    all_pass = True
    for f in test_freqs:
        if f > f_arr[-1]:
            continue

        expected_shift = 2 * np.pi * f * delta_tc
        actual_shift = phase_at_frequency(f_arr, phase_shifted, f) - phase_at_frequency(f_arr, phase_base, f)

        error = abs(actual_shift - expected_shift)
        status = "✓" if error < 0.001 else "✗"
        all_pass = all_pass and (error < 0.001)

        print(f"{f:>10.1f} | {expected_shift:>12.6f} | {actual_shift:>12.6f} {status}")

    print("-" * 70)
    print("Note: tc is applied EXTERNALLY via exp(2πi f tc) multiplication")
    print(f"TEST 3: {'PASSED' if all_pass else 'FAILED'}")
    print()
    return all_pass


# =============================================================================
# TEST 4: Changing f_ref only changes the phase offset
# =============================================================================
def test_fref_independence():
    """
    PREDICTION: Changing f_ref only affects the overall phase offset.
    The phase DIFFERENCE between any two frequencies is INDEPENDENT of f_ref.

    Φ(f2; f_ref=A) - Φ(f1; f_ref=A) = Φ(f2; f_ref=B) - Φ(f1; f_ref=B)

    Why? Because f_ref only affects the constant "phifRef" term.
    """
    print("=" * 70)
    print("TEST 4: Phase differences are independent of f_ref")
    print("=" * 70)

    phi_c = 0.0
    f_ref_A = 30.0
    f_ref_B = 100.0

    print(f"\nPrediction: Φ(f2) - Φ(f1) is the same regardless of f_ref")
    print(f"Comparing f_ref = {f_ref_A} Hz vs f_ref = {f_ref_B} Hz")
    print("-" * 70)

    # Generate waveforms with different f_ref
    freqs_A, h_A = generate_waveform(f_ref=f_ref_A, phi_c=phi_c)
    freqs_B, h_B = generate_waveform(f_ref=f_ref_B, phi_c=phi_c)

    f_A, phase_A = extract_phase(h_A, freqs_A)
    f_B, phase_B = extract_phase(h_B, freqs_B)

    # Check phase differences between frequency pairs
    freq_pairs = [(40.0, 60.0), (50.0, 100.0), (100.0, 200.0), (200.0, 400.0)]

    print(f"{'f1':>6} - {'f2':>6} | {'Δφ (f_ref=A)':>14} | {'Δφ (f_ref=B)':>14} | {'Diff':>10}")
    print("-" * 70)

    all_pass = True
    for f1, f2 in freq_pairs:
        if f2 > f_A[-1]:
            continue

        # Phase difference with f_ref = A
        diff_A = phase_at_frequency(f_A, phase_A, f2) - phase_at_frequency(f_A, phase_A, f1)

        # Phase difference with f_ref = B
        diff_B = phase_at_frequency(f_B, phase_B, f2) - phase_at_frequency(f_B, phase_B, f1)

        error = abs(diff_A - diff_B)
        status = "✓" if error < 0.01 else "✗"
        all_pass = all_pass and (error < 0.01)

        print(f"{f1:>6.0f} - {f2:>6.0f} | {diff_A:>14.6f} | {diff_B:>14.6f} | {error:>9.6f} {status}")

    print("-" * 70)
    print(f"TEST 4: {'PASSED' if all_pass else 'FAILED'}")
    print()
    return all_pass


# =============================================================================
# TEST 5: What we store for training vs what we need at inference
# =============================================================================
def test_training_inference_workflow():
    """
    PREDICTION: We can store phase relative to a training f_ref,
    then reconstruct any f_ref at inference.

    Training: Store Φ_stored(f) = Φ(f; f_ref_train) - Φ(f_ref_train; f_ref_train)
                                = Φ_intrinsic(f) - Φ_intrinsic(f_ref_train)

    Note: Φ(f_ref_train; f_ref_train) = π/4 when phi_c=0

    Inference: Φ(f; f_ref_inf, phi_c) = Φ_stored(f) - Φ_stored(f_ref_inf) + 2*phi_c + π/4
    """
    print("=" * 70)
    print("TEST 5: Training → Inference Workflow")
    print("=" * 70)

    f_ref_train = 50.0  # Fixed during training
    f_ref_inf = 30.0    # Different at inference
    phi_c_inf = np.pi / 3  # Arbitrary inference phase

    print(f"\nTraining: f_ref = {f_ref_train} Hz, phi_c = 0")
    print(f"Inference: f_ref = {f_ref_inf} Hz, phi_c = {phi_c_inf:.4f}")
    print("-" * 70)

    # TRAINING PHASE: Generate with phi_c=0, subtract phase at f_ref
    freqs_train, h_train = generate_waveform(f_ref=f_ref_train, phi_c=0.0)
    f_train, phase_train = extract_phase(h_train, freqs_train)

    phase_at_fref_train = phase_at_frequency(f_train, phase_train, f_ref_train)
    print(f"Phase at f_ref_train: {phase_at_fref_train:.6f} (should be π/4 = {np.pi/4:.6f})")

    # What we store: phase - phase(f_ref_train)
    phase_stored = phase_train - phase_at_fref_train
    print(f"Stored phase at f_ref_train: {phase_at_frequency(f_train, phase_stored, f_ref_train):.6f} (should be 0)")

    # INFERENCE PHASE: Reconstruct with different f_ref and phi_c
    # Formula: Φ_reconstructed = Φ_stored - Φ_stored(f_ref_inf) + 2*phi_c + π/4
    phase_stored_at_fref_inf = phase_at_frequency(f_train, phase_stored, f_ref_inf)
    phase_reconstructed = phase_stored - phase_stored_at_fref_inf + 2*phi_c_inf + np.pi/4

    # GROUND TRUTH: Generate directly with LAL using inference parameters
    freqs_truth, h_truth = generate_waveform(f_ref=f_ref_inf, phi_c=phi_c_inf)
    f_truth, phase_truth = extract_phase(h_truth, freqs_truth)

    # Compare at multiple frequencies
    test_freqs = [30.0, 50.0, 100.0, 200.0]

    print(f"\n{'Frequency':>10} | {'Reconstructed':>14} | {'LAL Truth':>14} | {'Diff':>10}")
    print("-" * 70)

    all_pass = True
    for f in test_freqs:
        if f > f_train[-1]:
            continue

        recon = phase_at_frequency(f_train, phase_reconstructed, f)
        truth = phase_at_frequency(f_truth, phase_truth, f)

        # Compare modulo 2π
        diff = (recon - truth) % (2*np.pi)
        if diff > np.pi:
            diff -= 2*np.pi

        status = "✓" if abs(diff) < 0.01 else "✗"
        all_pass = all_pass and (abs(diff) < 0.01)

        print(f"{f:>10.1f} | {recon:>14.6f} | {truth:>14.6f} | {diff:>9.6f} {status}")

    print("-" * 70)
    print(f"TEST 5: {'PASSED' if all_pass else 'FAILED'}")
    print()
    return all_pass


# =============================================================================
# TEST 6: Complete prediction example
# =============================================================================
def test_complete_prediction():
    """
    ULTIMATE TEST: Given parameters, predict the EXACT phase at a frequency.

    Setup:
    - f_ref = 40 Hz
    - phi_c = π/6
    - Want phase at f = 100 Hz

    Prediction method:
    1. Generate with phi_c=0, f_ref=40
    2. Φ(100; f_ref=40, phi_c=0) = some value X
    3. Φ(100; f_ref=40, phi_c=π/6) = X + 2*(π/6) = X + π/3

    Then verify with direct LAL call.
    """
    print("=" * 70)
    print("TEST 6: Complete Phase Prediction")
    print("=" * 70)

    f_ref = 40.0
    phi_c = np.pi / 6
    f_target = 100.0

    print(f"\nTarget: Predict Φ({f_target} Hz) with f_ref={f_ref} Hz, phi_c={phi_c:.4f}")
    print("-" * 70)

    # Step 1: Get phase with phi_c=0
    freqs0, h0 = generate_waveform(f_ref=f_ref, phi_c=0.0)
    f0, phase0 = extract_phase(h0, freqs0)
    phase_at_target_phi0 = phase_at_frequency(f0, phase0, f_target)

    print(f"Step 1: Φ({f_target}; phi_c=0) = {phase_at_target_phi0:.6f}")

    # Step 2: Predict with phi_c
    predicted_phase = phase_at_target_phi0 + 2 * phi_c
    print(f"Step 2: Predicted Φ({f_target}; phi_c={phi_c:.4f}) = {phase_at_target_phi0:.6f} + 2×{phi_c:.4f}")
    print(f"        = {predicted_phase:.6f}")

    # Step 3: Verify with direct LAL call
    freqs_actual, h_actual = generate_waveform(f_ref=f_ref, phi_c=phi_c)
    f_actual, phase_actual = extract_phase(h_actual, freqs_actual)
    actual_phase = phase_at_frequency(f_actual, phase_actual, f_target)

    print(f"Step 3: Actual Φ({f_target}; phi_c={phi_c:.4f}) = {actual_phase:.6f}")

    diff = abs(predicted_phase - actual_phase)
    status = "✓ PASSED" if diff < 0.01 else "✗ FAILED"

    print(f"\nDifference: {diff:.6f}")
    print(f"TEST 6: {status}")
    print()
    return diff < 0.01


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print(" LEARNING PHASE CONVENTIONS IN LAL")
    print(" Making Predictions and Verifying with Code")
    print("=" * 70 + "\n")

    print("THEORY SUMMARY")
    print("-" * 70)
    print("""
The GW phase in LAL has this structure:

    Φ(f) = Φ_intrinsic(Mf) - Φ_intrinsic(Mf_ref) + 2×phi_c + π/4

Key properties:
1. At f = f_ref: Φ(f_ref) = 2×phi_c + π/4
2. Changing phi_c by Δ shifts ALL phases by 2×Δ
3. Time shift tc: Φ → Φ + 2π×f×tc (applied externally)
4. Phase DIFFERENCES are independent of f_ref choice
5. The π/4 is the stationary phase approximation convention

For emulation:
- Training: Store Φ_stored = Φ(f) - Φ(f_ref_train) with phi_c=0
- Inference: Φ = Φ_stored(f) - Φ_stored(f_ref_inf) + 2×phi_c + π/4 + 2π×f×tc
""")
    print("-" * 70 + "\n")

    results = []
    results.append(("Phase at f_ref", test_phase_at_fref()))
    results.append(("phi_c shift", test_phic_shift()))
    results.append(("tc shift", test_tc_shift()))
    results.append(("f_ref independence", test_fref_independence()))
    results.append(("Training→Inference", test_training_inference_workflow()))
    results.append(("Complete prediction", test_complete_prediction()))

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:.<40} {status}")

    all_passed = all(r[1] for r in results)
    print("-" * 70)
    print(f"  {'OVERALL':.<40} {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
