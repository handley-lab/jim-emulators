#!/usr/bin/env python3
"""
Investigate the π offset in LAL phase conventions.

The tests show that the phase at f_ref is off by exactly π from our prediction.
Let's figure out why.
"""

import numpy as np
import lalsimulation as lalsim
import lal

# Parameters
M_TOTAL = 30.0
Q = 1.5
CHI1 = 0.2
CHI2 = 0.1
D_L = 100.0
INCLINATION = 0.0
F_MIN = 20.0
F_MAX = 1024.0
DELTA_F = 0.125


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
    h_cross = hc.data.data
    n_samples = len(h_plus)
    frequencies = np.arange(n_samples) * DELTA_F

    return frequencies, h_plus, h_cross


def main():
    print("=" * 70)
    print(" INVESTIGATING THE π OFFSET")
    print("=" * 70)

    f_ref = 50.0
    phi_c = 0.0

    freqs, hp, hc = generate_waveform(f_ref, phi_c)

    # Find index at f_ref
    idx_ref = int(f_ref / DELTA_F)

    print(f"\nAt f_ref = {f_ref} Hz (index {idx_ref}):")
    print("-" * 70)

    # Look at the complex values
    hp_at_ref = hp[idx_ref]
    hc_at_ref = hc[idx_ref]

    print(f"h_plus  = {hp_at_ref}")
    print(f"h_cross = {hc_at_ref}")
    print(f"|h_plus|  = {np.abs(hp_at_ref):.6e}")
    print(f"|h_cross| = {np.abs(hc_at_ref):.6e}")

    # Phase from h_plus
    phase_hp = np.angle(hp_at_ref)
    print(f"\nPhase of h_plus: {phase_hp:.6f} rad = {np.degrees(phase_hp):.2f}°")

    # Phase from h_cross
    phase_hc = np.angle(hc_at_ref)
    print(f"Phase of h_cross: {phase_hc:.6f} rad = {np.degrees(phase_hc):.2f}°")

    # Difference between hp and hc phases
    phase_diff = phase_hc - phase_hp
    print(f"h_cross - h_plus phase: {phase_diff:.6f} rad = {np.degrees(phase_diff):.2f}°")

    # For face-on (ι=0), h_cross should be 0 or hp and hc should be π/2 out of phase
    print("\n" + "=" * 70)
    print(" UNDERSTANDING THE CONVENTION")
    print("=" * 70)

    print("""
The GW strain in the FD is typically written as:

    h̃(f) = A(f) × exp(-i Ψ(f))      [note the MINUS sign]

The phase Ψ(f) increases with frequency (positive chirp).

When we compute np.angle(h), we get:
    np.angle(h) = -Ψ(f)   [wrapped to [-π, π]]

So our "measured phase" has the OPPOSITE sign from the physical phase!

If LAL sets Ψ(f_ref) = -2φ_c - π/4, then:
    np.angle(h(f_ref)) = -Ψ(f_ref) = 2φ_c + π/4

Let's check...
""")

    # Check if the convention is Ψ = -2φ_c - π/4 or +2φ_c + π/4
    print(f"With phi_c = 0:")
    print(f"  If Ψ(f_ref) = -π/4, then angle(h) = +π/4 = {np.pi/4:.4f}")
    print(f"  If Ψ(f_ref) = +π/4, then angle(h) = -π/4 = {-np.pi/4:.4f}")
    print(f"  Actual angle(h_plus) at f_ref = {phase_hp:.4f}")

    # Now let's look at the UNWRAPPED phase near f_ref
    print("\n" + "=" * 70)
    print(" UNWRAPPED PHASE NEAR f_ref")
    print("=" * 70)

    mask = (freqs >= F_MIN) & (np.abs(hp) > 0)
    phase_unwrapped = np.unwrap(np.angle(hp[mask]))
    freqs_valid = freqs[mask]

    # Find phase at f_ref
    idx_in_valid = np.searchsorted(freqs_valid, f_ref)
    phase_at_fref = phase_unwrapped[idx_in_valid]

    print(f"\nUnwrapped phase at f_ref = {f_ref} Hz:")
    print(f"  Phase = {phase_at_fref:.4f} rad")
    print(f"  Phase mod 2π = {phase_at_fref % (2*np.pi):.4f} rad")
    print(f"  Cycles from f_min to f_ref = {(phase_at_fref - phase_unwrapped[0]) / (2*np.pi):.2f}")

    # The key insight: the WRAPPED phase at f_ref should tell us about the convention
    print("\n" + "=" * 70)
    print(" THE KEY INSIGHT")
    print("=" * 70)

    # Check at different phi_c values
    print(f"\nWrapped phase at f_ref for different phi_c values:")
    print(f"{'phi_c':>10} | {'Expected':>12} | {'angle(hp)':>12} | {'Diff':>10}")
    print("-" * 50)

    for phi_c_test in [0, np.pi/4, np.pi/2, np.pi]:
        freqs_t, hp_t, _ = generate_waveform(f_ref, phi_c_test)
        idx = int(f_ref / DELTA_F)
        actual = np.angle(hp_t[idx])

        # Two possible conventions:
        expected_plus = (2*phi_c_test + np.pi/4) % (2*np.pi)
        if expected_plus > np.pi:
            expected_plus -= 2*np.pi

        expected_minus = (-2*phi_c_test - np.pi/4) % (2*np.pi)
        if expected_minus > np.pi:
            expected_minus -= 2*np.pi

        diff_plus = actual - expected_plus
        diff_minus = actual - expected_minus

        if abs(diff_plus) < abs(diff_minus):
            expected = expected_plus
            diff = diff_plus
        else:
            expected = expected_minus
            diff = diff_minus

        print(f"{phi_c_test:>10.4f} | {expected:>12.4f} | {actual:>12.4f} | {diff:>10.4f}")

    print("\n" + "=" * 70)
    print(" TESTING SIGN CONVENTION")
    print("=" * 70)

    # Generate with phi_c = π/4
    freqs1, hp1, _ = generate_waveform(f_ref, 0.0)
    freqs2, hp2, _ = generate_waveform(f_ref, np.pi/4)

    idx = int(f_ref / DELTA_F)
    angle1 = np.angle(hp1[idx])
    angle2 = np.angle(hp2[idx])

    print(f"\nphi_c = 0:     angle(hp) at f_ref = {angle1:.6f}")
    print(f"phi_c = π/4:   angle(hp) at f_ref = {angle2:.6f}")
    print(f"Difference: {angle2 - angle1:.6f}")
    print(f"Expected if 2×phi_c: {2*np.pi/4:.6f} = {np.pi/2:.6f}")

    # What about minus sign?
    print(f"\nIf h ~ exp(-iΨ) with Ψ = 2φ_c + stuff:")
    print(f"  angle(h) = -Ψ = -2φ_c - stuff")
    print(f"  Changing φ_c by +π/4 changes angle by -π/2 = {-np.pi/2:.4f}")
    print(f"  Actual change: {angle2 - angle1:.4f}")

    print("\n" + "=" * 70)
    print(" CONCLUSION")
    print("=" * 70)

    # Check if increasing phi_c INcreases or DEcreases angle
    if angle2 > angle1:
        print("""
The phase INCREASES with phi_c, so the convention is:

    h(f) ~ exp(+i × Φ)  where Φ = 2×phi_c + ...

NOT the usual exp(-i×Ψ) convention!

This means our prediction formula should be:
    Φ(f_ref) = 2×phi_c + some_constant

where some_constant ≠ π/4.
""")
    else:
        print("""
The phase DECREASES with phi_c, so the convention is:

    h(f) ~ exp(-i × Ψ)  where Ψ involves phi_c

This is the standard convention, but the offset is different from π/4.
""")

    # Let's figure out the actual constant
    print("\n" + "=" * 70)
    print(" DETERMINING THE ACTUAL OFFSET")
    print("=" * 70)

    # With phi_c = 0, what is angle(hp) at f_ref?
    actual_offset = angle1  # This is angle(hp) at f_ref when phi_c=0
    print(f"\nWith phi_c = 0, angle(hp) at f_ref = {actual_offset:.6f} rad")
    print(f"This equals: {actual_offset/np.pi:.4f}π")

    # Check if this matches -π/4 or +3π/4 or something else
    candidates = [
        ("π/4", np.pi/4),
        ("-π/4", -np.pi/4),
        ("3π/4", 3*np.pi/4),
        ("-3π/4", -3*np.pi/4),
        ("5π/4 - 2π = -3π/4", -3*np.pi/4),
    ]

    print(f"\nComparing to standard offsets:")
    for name, value in candidates:
        diff = abs(actual_offset - value)
        if diff > np.pi:
            diff = 2*np.pi - diff
        print(f"  {name:>20}: diff = {diff:.6f}")

    # THE ANSWER
    print("\n" + "=" * 70)
    print(" THE FORMULA")
    print("=" * 70)
    print(f"""
Based on the analysis, at f_ref with phi_c=0:
    angle(h_plus) = {actual_offset:.6f} rad ≈ {actual_offset/np.pi:.2f}π

The relationship appears to be:
    angle(h_plus at f_ref) = 2×phi_c + {actual_offset:.4f}

For our tests, the π offset came from the difference:
    Expected: π/4 = {np.pi/4:.4f}
    Actual offset: {actual_offset:.4f}
    Difference: {actual_offset - np.pi/4:.4f} ≈ {(actual_offset - np.pi/4)/np.pi:.2f}π
""")


if __name__ == "__main__":
    main()
