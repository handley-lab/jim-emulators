# Phase Reference Frequency Conventions in LAL

This document provides a deep dive into how phase reference frequencies work in LALSimulation, specifically for IMRPhenomX waveforms. Understanding these conventions is critical for ensuring trained emulators can be correctly applied at inference time.

## Executive Summary

**Key findings (VERIFIED BY TESTS):**

1. The GW phase has both **intrinsic** (parameter-dependent) and **extrinsic** (reference-dependent) components
2. LAL applies a **time shift** (`linb*Mf`) to align the waveform peak at t≈0
3. **CRITICAL**: Different LAL functions have different phase conventions!
4. For emulation, we learn the intrinsic phase shape and apply extrinsic corrections at inference

### Phase Convention Summary

| LAL Function | Use Case | Phase at f_ref (phi_c=0) | phi_c effect |
|--------------|----------|--------------------------|--------------|
| `SimIMRPhenomXHM` | Full waveform (h+, h×) | -3π/4 | +2×Δphi_c |
| `SimIMRPhenomXHMFrequencySequenceOneMode` | Individual mode h_lm | **-π/4** | **-2×Δphi_c** |

**Our training uses `FrequencySequenceOneMode`**, so the inference formula has a **negative** phi_c term!

## Detailed Analysis

### 1. LAL Phase Decomposition

For the (2,2) mode, LAL computes the phase as:

```
Φ(f) = (1/η) × Φ_raw(Mf) + linb×Mf + lina + phifRef
```

Where:
- `Φ_raw(Mf)` is the raw phase model (PN + intermediate + ringdown)
- `linb` is the time shift that aligns the waveform peak to t≈0
- `lina` is typically 0
- `phifRef` is the reference phase correction

### 2. Time Shift Calculation (linb)

The time shift is computed in `IMRPhenomX_TimeShift_22()`:

```c
// Fit of dΦ/df at f_ring - f_damp
linb = XLALSimIMRPhenomXLinb(eta, STotR, dchi, delta);

// Phase derivative at the ringdown reference point
dphi22Ref = (1/eta) * IMRPhenomX_dPhase_22(f_ring - f_damp, ...);

// Correction for peak alignment (500M before end, plus psi4→strain)
tshift = linb - dphi22Ref - 2π × (500 + psi4tostrain);
```

**Key point**: The time shift depends only on intrinsic parameters (η, χ₁, χ₂), NOT on f_ref.

### 3. Reference Phase Calculation (phifRef)

The reference phase is computed as:

```c
phifRef = -[(1/η)×Φ_raw(Mf_ref) + linb×Mf_ref + lina] + 2×phi0 + π/4
```

This ensures that at f = f_ref:
```
Φ(f_ref) = 2×phi0 + π/4
```

The factor of 2 comes from the orbital→GW phase relationship (quadrupole: f_GW = 2×f_orbit).
The π/4 is a convention from the stationary phase approximation.

### 4. Physical Interpretation

The complete phase at any frequency is:

```
Φ(f) = Φ_intrinsic(Mf) - Φ_intrinsic(Mf_ref) + 2×phi0 + π/4

where Φ_intrinsic(Mf) = (1/η)×Φ_raw(Mf) + linb×Mf + lina
```

This means:
- **Φ_intrinsic(Mf)** is the intrinsic phase evolution (parameter-dependent, f_ref-independent)
- The term `- Φ_intrinsic(Mf_ref) + 2×phi0 + π/4` is an overall constant shift

### 5. Higher Modes

For higher modes (ℓ,m) ≠ (2,2):

- The phase is related to the (2,2) mode by: `Φ_lm(f) ≈ (m/2) × Φ_22((2/m) × f) + δΦ_lm`
- Each mode has its own phase shift `deltaphiLM` computed at alignment frequency
- The relationship to orbital phase is `Φ_lm = m × Φ_orbital`

For positive vs negative m:
```c
if (emm > 0) {
    phase = addpi - phase;  // addpi = π if l is odd, 0 if even
}
```

This implements conjugate symmetry: h_{l,-m}(f) = (-1)^ℓ × h*_{lm}(f)

## Implications for Emulation

### Current Training Approach

Our current training (`scripts/generate_data.py`) does:

1. Call LAL with `phase=0` (phi0=0) and `f_ref = f_ref_train` (fixed)
2. LAL returns phase that's ≈ π/4 at f_ref (due to phifRef)
3. Subtract phase at f_ref to make it exactly 0

Result: We learn `Φ_stored(Mf) = Φ_intrinsic(Mf) - Φ_intrinsic(Mf_ref_train)`

### Inference Procedure

At inference time, to reconstruct the full waveform with arbitrary f_ref and phi_c:

```python
# 1. Evaluate emulator at both target frequencies and inference f_ref
phase_emu_at_f = emulator.predict_phase(Mf)           # Array
phase_emu_at_ref = emulator.predict_phase(Mf_ref_inference)  # Scalar

# 2. Reconstruct phase with correct offset
phase_reconstructed = phase_emu_at_f - phase_emu_at_ref + 2*phi_c + np.pi/4

# 3. Apply time-of-coalescence shift
phase_final = phase_reconstructed + 2*np.pi*f*tc
```

This correctly handles:
- Different f_ref values at inference vs training
- Arbitrary coalescence phase phi_c
- Time-of-coalescence shift tc

### Alternative: Store Intrinsic Phase Directly

For more flexibility, could modify training to store `Φ_intrinsic(Mf)` directly:

```python
# Don't subtract phase at reference
phase_intrinsic = np.unwrap(np.angle(h_lm))  # No alignment
```

Then at inference:
```python
phase = phase_intrinsic(Mf) - phase_intrinsic(Mf_ref) + 2*phi_c + np.pi/4
```

**Tradeoff**: The current approach (subtracting at reference) makes the phase numerically smaller and better conditioned for PCA, but requires knowing Mf_ref_train at inference.

## Verification Tests

To verify correct phase handling, check:

1. **Self-consistency**: Generate LAL waveform with phi0=0, f_ref=f1. The phase at f1 should be π/4.

2. **Phase offset**: Generate with phi0=φ, f_ref=f1. The phase at f1 should be 2φ + π/4.

3. **f_ref independence**: The phase *difference* between two frequencies should be independent of f_ref (only the absolute offset changes).

4. **Emulator reconstruction**: For a validation sample, check that:
   ```
   emulator_phase(Mf) - emulator_phase(Mf_ref) + π/4 ≈ LAL_phase(Mf)
   ```
   with the same intrinsic parameters.

## Key Code References

- `LALSimIMRPhenomX_internals.c:2624` - `IMRPhenomX_TimeShift_22()`
- `LALSimIMRPhenomX_internals.c:2827-2849` - Phase reconstruction with phifRef
- `LALSimIMRPhenomXHM.c:2714-2796` - Single mode phase calculation
- `LALSimIMRPhenomXHM_internals.c:2363-2368` - Higher mode phase alignment
- `ripple/src/ripplegw/waveforms/IMRPhenomXAS.py:1343-1357` - JAX implementation

## Summary Table

| Component | Formula | Depends on f_ref? | Learned by NN? |
|-----------|---------|-------------------|----------------|
| Φ_raw(Mf) | PN + intermediate + RD | No | Yes (indirectly) |
| linb (time shift) | Fit at f_ring - f_damp | No | Yes (included) |
| phifRef | -Φ_intrinsic(Mf_ref) + 2φ₀ + π/4 | Yes | No (applied at inference) |
| 2πf×tc | Time of coalescence | No | No (applied at inference) |

## Recommendations

1. **Keep current training approach**: Subtracting phase at reference is numerically well-conditioned

2. **Store Mf_ref_train in model metadata**: The inference code needs to know what reference was used in training (already done in generate_data.py via `f.attrs["Mf_ref"]`)

3. **Provide inference helper for individual modes**:
   ```python
   # For individual modes (SimIMRPhenomXHMFrequencySequenceOneMode)
   # Generalized for any (l, m) mode
   PHASE_OFFSET_MODE = -np.pi / 4  # For most m > 0 modes

   def apply_mode_phase_conventions(phase_stored, f, f_ref_inference, phi_c, tc,
                                     f_train_grid, m=2):
       """Apply LAL-compatible phase conventions to emulator output for individual modes.

       Args:
           phase_stored: Emulator output (phase - phase(f_ref_train))
           f: Frequency array in Hz
           f_ref_inference: Reference frequency for this waveform (Hz)
           phi_c: Coalescence phase (radians)
           tc: Time of coalescence (seconds)
           f_train_grid: Frequency grid used during training
           m: Azimuthal mode number (default 2 for dominant mode)

       Returns:
           Phase array matching LAL FrequencySequenceOneMode output
       """
       phase_at_ref = np.interp(f_ref_inference, f_train_grid, phase_stored)
       # NOTE: -m*phi_c for positive m modes (generalized from -2*phi_c)
       # NOTE: -2πf*tc (MINUS sign from Fourier shift theorem!)
       phase = phase_stored - phase_at_ref - m * phi_c + PHASE_OFFSET_MODE
       if tc != 0:
           phase -= 2 * np.pi * f * tc  # MINUS sign!
       return phase
   ```

4. **For full waveform reconstruction** (combining modes):

   When combining modes to get h+ and h×, use the standard formula:
   ```python
   # h+ - i*h× = Σ_{l,m} h_lm(f) × Y_{lm}(ι, φ)
   # where φ is the observer's azimuthal angle (NOT phi_c!)

   def reconstruct_polarizations(mode_dict, iota, phi_observer, f, f_ref, phi_c, tc):
       """Combine modes into h+ and h× polarizations.

       Args:
           mode_dict: Dict mapping (l,m) to complex h_lm arrays
           iota: Inclination angle (radians)
           phi_observer: Observer azimuthal angle (radians) - geometric, not phi_c!
           f: Frequency array
           ...
       """
       from scipy.special import sph_harm

       h_complex = np.zeros_like(f, dtype=complex)
       for (ell, m), h_lm in mode_dict.items():
           # Spin-weighted spherical harmonic (s=-2)
           Y_lm = spin_weighted_spherical_harmonic(-2, ell, m, iota, phi_observer)
           h_complex += h_lm * Y_lm

       h_plus = h_complex.real
       h_cross = -h_complex.imag
       return h_plus, h_cross
   ```

5. **Key insight from Gemini review**: The difference between individual modes and full waveform
   comes from separating **physical** parameters (phi_c = coalescence phase) from **geometric**
   parameters (φ = observer azimuthal angle). The spherical harmonics Y_lm contain e^{imφ},
   which is why the full waveform sees `+m×φ` while individual modes see `-m×phi_c`.

6. **Validate against LAL**: Always test that reconstructed waveforms match LAL to target mismatch

7. **Negative m modes**: For non-precessing signals, `h_{l,-m} = (-1)^l × h*_{lm}`.
   The phi_c dependence for m<0 is `+|m|×phi_c` (opposite sign).

## Verification Scripts

The following scripts verify the phase conventions:

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/learn_phase_conventions.py` | Full waveform tests (original) | Some fail |
| `scripts/learn_phase_conventions_v2.py` | Full waveform with -3π/4 correction | All pass |
| `scripts/investigate_pi_offset.py` | Investigation of full waveform offset | Complete |
| `scripts/verify_mode_phase_convention.py` | Compare full vs individual mode | Shows difference |
| `scripts/investigate_mode_convention.py` | Individual mode convention | All pass |

Run the individual mode tests (matches generate_data.py):
```bash
mamba activate jim-emulator
python scripts/investigate_mode_convention.py
```
