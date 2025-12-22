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

## Implementation

### JAX/Flax Neural Network (`snippets/flax_nn.py`)

Reference implementation for Speculator-style emulator:
- **SpeculatorActivation**: σ(x) = [γ + sigmoid(αx)·(1-γ)]·x with γ ∈ (0,1) via sigmoid constraint
- **EmulatorMLP**: 4 hidden layers × 512 units with Speculator activation
- **WaveformPCA**: JAX-compatible PCA for waveform compression
- **GWEmulator**: Full pipeline: params → NN → PCA → amplitude/phase

Key design decisions (see `docs/jax-flax-best-practices.md`):
- gamma_logits initialized to -2.0 (mostly gated) - empirically outperforms linear start
- Shape inference (no explicit features argument)
- Gradient clipping with `optax.clip_by_global_norm(1.0)`
- `jax_enable_x64` set only at process start, not in library modules

### Training Scripts (`scripts/`)

- `train_toy_example.py`: Verifies training pipeline with synthetic data (1330x loss improvement)

### Transcripts (`transcripts/`)

Meeting transcripts documenting planning discussions and decisions:

| File | Content |
|------|---------|
| `2025-12-22-09-07-00.txt` | Initial planning: project goals, parameterization (Mf, η, χ), domain choices, CosmoPower/Speculator approach |
| `2025-12-22-12-10-00.txt` | Progress review: LAL wrapper, sensitivity plots, mode additivity discovery, per-mode emulation strategy |
| `2025-12-22-13-33-00.txt` | Strategy consolidation: start with (2,2) mode, aligned spins; literature gap analysis; mlgw_NN paper discovery |

**Key insights from transcripts:**
- Waveforms depend on Mf (mass × frequency), enabling mass-independent training
- 7D maximum parameter space: η + 6 spin components; aligned-spin reduces to 3D
- Modes are additive to machine precision - emulate per-mode then sum
- Individual modes are much smoother than full waveform (removes interference wiggles)
- Inclination can be factored out via explicit spherical harmonics (from mlgw_NN)
- Start with (2,2) mode aligned-spin case, build up mode by mode

## Deep Research Literature

See `docs/literature/deep-research/` for survey of advanced NN architectures for GW emulation.

**Key papers for our approach:**

| ArXiv ID | Paper | Relevance |
|----------|-------|-----------|
| `2205.14066/` | **SEOBNN**: NN surrogate for SEOBNRv4PHM | **Key reference** - precessing + HM, co-precessing frame decomposition |
| `2402.06587/` | **mlgw_NN**: Time-domain GW with HM | Amp/phase → PCA → NN, 10⁻⁴ mismatch |
| `2408.02470/` | **obiwann**: SVD-NN interpolant | Validates SVD+MLP approach (10⁻⁴ mismatch) |
| `2512.02968/` | **Dingo-T1**: Transformers for GW PE | Uses IMRPhenomXPHM - validates our target model |

**Key background (2205.14066 SEOBNN)**: Closest existing work to our goal. Template for our IMRPhenomXPHM emulator.

**Coprecessing frame technique (from SEOBNN):**
1. Transform to frame tracking orbital plane → waveform simplifies
2. Model separately: coprecessing modes (smooth) + Euler angles α(t), β(t), γ(t)
3. SVD for dimensionality reduction
4. Neural networks predict coefficients at empirical interpolation nodes
5. Rotate back to inertial frame for final waveform

**Performance targets** (based on SEOBNN benchmarks):
- CPU: ~18 ms/waveform (100-200× speedup)
- GPU batch: <0.01 ms/waveform (>100,000× speedup)

**Design principle**: "Bake domain knowledge into architecture" - use physics-informed decomposition (coprecessing frame, amp/phase split) rather than learning raw physics.

**Training data insight**: All papers train on semi-analytical approximants (SEOBNR, IMRPhenom, EOB), not raw NR. This validates our approach of emulating IMRPhenomXPHM directly.

## Theory Document

**`theory/general-waveform-decomposition.tex`** (12 pages, approved)

Comprehensive framework for NN emulation of arbitrary GW waveforms:
- General decomposition: spherical harmonic → coprecessing frame → amplitude/phase
- Coordinate choices vs restrictive assumptions (conjugate symmetry, circular orbits)
- Three architecture options with Option 3 (end-to-end learning) recommended
- Regularization addressing gauge non-identifiability: L_mode + L_angle + L_minrot + L_anchor
- Initialization strategy for training stability

**Key design decisions:**
- Regularize instantaneous frequency (ω̈), not raw phase (φ̈) - avoids fighting physical chirp
- Penalize angle velocities (α̇, β̇, γ̇) to prevent stealing orbital frequency
- Anchor frame at reference time: α(t_ref) = γ(t_ref) = 0
- Use sin/cos representation for angles to avoid 2π discontinuities

**Citations:** Blackman (2015/2017), Varma (2019), Boyle (2011), Schmidt (2011)
