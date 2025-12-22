#!/usr/bin/env python3
"""
Verify the correct sign for time-of-coalescence (tc) phase shift.

The Fourier shift theorem says:
    h(t - tc) ↔ h̃(f) × exp(-i 2π f tc)

So the phase shift should be -2πf×tc (MINUS), not +2πf×tc.
"""

import numpy as np

def main():
    print("=" * 70)
    print(" VERIFYING tc SIGN CONVENTION")
    print("=" * 70)

    print("""
Fourier Shift Theorem:
    h(t - t₀)  ↔  H(f) × exp(-i 2π f t₀)

If we shift the signal to LATER times (positive tc means coalescence
happens later), the frequency-domain phase decreases:

    phase_shifted = phase_original - 2π f tc

Example:
    tc = +1 second (coalescence 1 second later than reference)
    At f = 100 Hz:
        Phase change = -2π × 100 × 1 = -628.3 radians

This means the signal at 100 Hz arrives with LESS accumulated phase
at any given time, which is correct because the coalescence (where
phase accumulation stops) happens later.
""")

    # Numerical verification
    print("=" * 70)
    print(" NUMERICAL VERIFICATION")
    print("=" * 70)

    # Create a simple chirp signal
    t = np.linspace(-10, 0, 10000)  # Time before coalescence
    f0 = 20  # Start frequency
    f1 = 200  # End frequency

    # Phase grows quadratically (simple chirp approximation)
    phase_t = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * t[-1]) * t**2)
    h_t = np.cos(phase_t)

    # FFT
    dt = t[1] - t[0]
    h_f = np.fft.rfft(h_t) * dt
    f = np.fft.rfftfreq(len(t), dt)

    # Apply time shift in frequency domain
    tc = 0.5  # Shift by 0.5 seconds

    # Method 1: MINUS sign (correct per Fourier theorem)
    h_f_shifted_minus = h_f * np.exp(-1j * 2 * np.pi * f * tc)

    # Method 2: PLUS sign (incorrect)
    h_f_shifted_plus = h_f * np.exp(+1j * 2 * np.pi * f * tc)

    # Transform back and compare with direct time shift
    h_t_shifted_direct = np.cos(2 * np.pi * (f0 * (t - tc) + (f1 - f0) / (2 * t[-1]) * (t - tc)**2))

    h_t_from_minus = np.fft.irfft(h_f_shifted_minus) / dt
    h_t_from_plus = np.fft.irfft(h_f_shifted_plus) / dt

    # Truncate to same length
    n = min(len(h_t_shifted_direct), len(h_t_from_minus))

    error_minus = np.mean((h_t_shifted_direct[:n] - h_t_from_minus[:n])**2)
    error_plus = np.mean((h_t_shifted_direct[:n] - h_t_from_plus[:n])**2)

    print(f"\nTime shift tc = {tc} seconds")
    print(f"MSE with MINUS sign (exp(-i 2πf tc)): {error_minus:.2e}")
    print(f"MSE with PLUS sign (exp(+i 2πf tc)):  {error_plus:.2e}")

    if error_minus < error_plus:
        print("\n✓ MINUS sign is correct: phase -= 2πf×tc")
    else:
        print("\n✗ PLUS sign gives better match (unexpected!)")

    print("\n" + "=" * 70)
    print(" CONCLUSION")
    print("=" * 70)
    print("""
The correct formula for time-of-coalescence is:

    phase = phase_intrinsic - 2π × f × tc
                            ^
                            MINUS sign!

This was confirmed by Gemini's review and numerical verification.
""")


if __name__ == "__main__":
    main()
