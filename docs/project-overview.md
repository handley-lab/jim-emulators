# Jim Emulators: Project Overview

## Goal

Emulate gravitational wave waveforms in JAX for integration into the jim ecosystem. The focus is on practical emulation following the CosmoPower/Speculator approach, not on novel neural network architectures.

## Physical Foundations

### Parameter Space Reduction

The key insight is to emulate only the **intrinsic** waveform parameters and handle extrinsic parameters analytically:

**Core parameters (7D maximum):**
- `eta` (symmetric mass ratio)
- `chi_1x, chi_1y, chi_1z` (spin components of first black hole)
- `chi_2x, chi_2y, chi_2z` (spin components of second black hole)

**Aligned-spin simplification (3D):**
- `eta, chi_1z, chi_2z`

This is the IMRPhenomXAS case - a good starting point.

**Handled analytically outside the emulator:**
- Distance: scales as `1/D_L`
- Reference phase: adds `phi_c`
- Inclination angles (`beta_jn`, `phi_jl`) are reparameterizations of the 6 spin components

### Frequency Scaling

Waveforms depend on `M * f` (mass times frequency), not `f` alone. This means:
- Train on a grid of `Mf` values
- Rescale to any mass analytically
- This scaling works naturally in the frequency domain
- In the time domain, the scaling gets "mixed up" - same morphology but different time grids

## Domain Choices

### Frequency Domain (Recommended for MVP)

Preferred because:
- `Mf` scaling can be done analytically
- Natural domain for IMRPhenom waveforms

Within frequency domain, two representation choices:
1. Real and imaginary parts
2. Amplitude and phase

The optimal choice should be determined empirically by examining smoothness.

### Time Domain

May be worth exploring (Alessio Spurio Mancini reportedly tried this), but complicates the mass scaling.

## Problem Specification

Understanding the numerical challenge:

- **Amplitude**: varies over ~5 orders of magnitude
- **Phase**: varies by ~100-200 radians
- **Frequency range**: ~0.001 to ~0.1-0.2 (dynamic range ~10^3)
- **Target accuracy**: mismatches at level ~10^-3

This specification is useful for consulting ML literature outside astronomy.

## Development Approach

### Phase 1: Training Data Generation

1. Install LALSuite: `pip install lalsuite`
2. Generate waveforms using LAL (see ripple examples for calling conventions)
3. Store as **complex strain** - this allows flexibility to later work with:
   - Amplitude and phase
   - Real and imaginary parts
   - Time domain (via FFT)

### Phase 2: Basic Emulator Validation

Before jumping into GW waveforms, validate the approach:
- The CosmoPower/Speculator framework involves non-trivial PCA
- Consider testing on a simpler problem first
- Or attempt to reproduce existing results

### Phase 3: Parameter Sensitivity Analysis

1. **Parameter sensitivity plots**: Run IMRPhenomXPHM on a grid, varying one parameter at a time. Examine:
   - Smoothness of amplitude/phase variations
   - Dynamic ranges
   - Which representation (amp/phase vs real/imag) is smoother

2. **Reference plots**: Create baseline visualizations for the project

3. **Benchmarking**: Use mismatch calculations (see ripple paper 2306.17245) to validate emulator accuracy

### Future: External Review

Once benchmarks are established, consider canvassing ML literature outside astronomy for alternative approaches. Having a well-specified problem makes it easier to get external input.

## Key Resources

### Papers
- **CosmoPower** (2106.03846): Neural network emulation approach
- **Speculator** (1911.11778): Same underlying methodology
- **ripple** (2302.05329): JAX waveforms, mismatch calculations
- **ripple recalibration** (2306.17245): Waveform comparison methodology
- **IMRPhenomXPHM** (2004.06503): Precessing higher modes waveform
- **IMRPhenomXHM** (2001.10914): Non-precessing higher modes

### Code
- `cosmopower/`: Neural network emulator architecture
- `ripple/`: JAX waveform implementation, comparison scripts
- `lalsuite/lalsimulation/`: Reference LAL waveform implementations

## Architecture Notes

The ripple repository contains:
- Mismatch calculation scripts
- LAL calling conventions
- Phase conventions
- Test infrastructure for waveform comparison

These can be adapted for validating the emulator against LAL reference waveforms.
