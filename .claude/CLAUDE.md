# Jim Emulators Project

**Repository**: https://github.com/handley-lab/jim-emulators

**Goal**: Emulate GW waveforms in JAX for the jim ecosystem, following CosmoPower/Speculator approach.

**Documentation**: See `docs/project-overview.md` for full project details.

## Development Log

**IMPORTANT**: Maintain `docs/development-log.md` as work progresses. This project aims to publish both the emulators and the AI-assisted workflow used to create them.

When making significant progress:
1. Update the development log with narrative descriptions of work done
2. Include specific commands, file paths, and tool usage
3. Document decisions made and rationale
4. Record any insights or lessons learned
5. Keep the "Current State" and "Next Steps" sections up to date

The log should be more narrative than git commits - it tells the story of how the project was built.

## Reference Papers

All paper sources downloaded to project root.

### Cosmological Emulators

| ArXiv ID | Paper | Main File |
|----------|-------|-----------|
| `2106.03846/` | **CosmoPower**: emulating cosmological power spectra for accelerated Bayesian inference | `CosmoPower.tex` |
| `1911.11778/` | **Speculator**: Emulating stellar population synthesis for fast and accurate galaxy spectra and photometry | `emulator.tex` |

### Gravitational Wave Waveforms

| ArXiv ID | Paper | Main File |
|----------|-------|-----------|
| `2302.05329/` | **ripple**: Differentiable and Hardware-Accelerated Waveforms for GW Data Analysis | `ms.tex` |
| `2306.17245/` | Recalibrating GW Phenomenological Waveform Model (ripple mismatch calculations) | `ms.tex` |
| `2004.06503/` | **IMRPhenomXPHM**: Computationally efficient models for precessing BBH with higher modes | `phenomx.tex` |
| `2001.10914/` | **IMRPhenomXHM**: Multi-mode frequency-domain model for non-precessing BBH | `hmc2.tex` |

### Key Authors

- **CosmoPower**: A. Spurio Mancini, D. Piras, J. Alsing
- **Speculator**: Justin Alsing, Hiranya Peiris, Joel Leja
- **ripple**: Thomas D. P. Edwards, Kaze W. K. Wong, Kelvin K. H. Lam
- **IMRPhenomX**: Geraint Pratten, Cecilio García-Quirós, Marta Colleoni, Sascha Husa

## Code Repositories

| Directory | Repository | Description |
|-----------|------------|-------------|
| `cosmopower/` | `git@github.com:alessiospuriomancini/cosmopower.git` | Neural network emulator for cosmological power spectra |
| `ripple/` | `git@github.com:tedwards2412/ripple.git` | JAX-based differentiable GW waveforms |
| `lalsuite/lalsimulation/` | `https://git.ligo.org/lscsoft/lalsuite.git` (sparse) | LIGO waveform implementations (IMRPhenomX etc.)

## Theory Documents

### Frequency Domain (Immediate Target)

**`theory/frequency-domain-emulation.tex`** (14 pages)

Framework for emulating IMRPhenomXHM (aligned-spin, higher modes):
- Mass scaling: network learns H_ℓm(Mf; η, χ₁, χ₂), total mass M is not an intrinsic parameter
- Extrinsic parameters (D_L, t_c, φ_c, ι) enter as analytic multiplicative factors
- Conjugate symmetry halves mode count: only need m > 0 modes
- Representation: log-amplitude, unwrapped phase, log-spaced Mf grid
- Architecture: PCA + MLP (CosmoPower-style) recommended

### Time Domain (Future: Precessing/Eccentric)

**`theory/general-waveform-decomposition.tex`** (12 pages, approved)

Comprehensive framework for arbitrary GW waveforms including precession:
- General decomposition: spherical harmonic → coprecessing frame → amplitude/phase
- Coordinate choices vs restrictive assumptions (conjugate symmetry, circular orbits)
- Three architecture options with Option 3 (end-to-end learning) recommended
- Regularization addressing gauge non-identifiability: L_mode + L_angle + L_minrot + L_anchor

**Key design decisions (time domain):**
- Regularize instantaneous frequency (ω̈), not raw phase (φ̈) - avoids fighting physical chirp
- Penalize angle velocities (α̇, β̇, γ̇) to prevent stealing orbital frequency
- Anchor frame at reference time: α(t_ref) = γ(t_ref) = 0
- Use sin/cos representation for angles to avoid 2π discontinuities

**Citations:** Blackman (2015/2017), Varma (2019), Boyle (2011), Schmidt (2011)
