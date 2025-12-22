# Jim Emulators Project

**Repository**: https://github.com/handley-lab/jim-emulators

**Goal**: Emulate GW waveforms in JAX for the jim ecosystem, following CosmoPower/Speculator approach.

**Documentation**: See `docs/project-overview.md` for full project details.

## Why This Project Exists

Nobody else is doing frequency-domain, aligned-spin, higher-mode emulation with PCA + MLP. The literature focuses on harder problems:

| Paper | Domain | Physics | Why Different |
|-------|--------|---------|---------------|
| Grimbergen (2402.06587) | Time | HOMs | TD not FD; uses SEOBNRv4HM |
| Thomas (2205.14066) | Time | Precessing | TD; precessing focus |
| Shi (2411.14893) | Time | Eccentric | TD; eccentric focus |
| gwharmone (2504.12420) | Frequency | Eccentric | Eccentric, uses GPR |
| Chase Orbits (2510.00116) | Both | Eccentric | Eccentric focus |

**The gap exists because IMRPhenomXHM is already fast** — it's an analytic phenomenological model, not an ODE solver like EOB. People don't emulate something that's already O(ms).

**Why we're doing it anyway:**

1. **JAX differentiability** — IMRPhenomXHM exists in LAL (C/Python), not JAX. We need gradients for jim's HMC sampler.

2. **Faster than hand-porting** — ripple rewrote IMRPhenomD in JAX by hand. We're testing whether emulation is a faster path to get more models (XHM, XPHM) into JAX.

3. **Proof of concept** — if PCA + MLP works for the "easy" aligned-spin case, the same infrastructure extends to precessing/eccentric.

**We're filling a gap that exists because of different motivations (JAX ecosystem) rather than because the problem is unsolved.**

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

### Waveform Surrogates (from Deep Research)

| ArXiv ID | Paper | Main File |
|----------|-------|-----------|
| `2402.06587/` | **Grimbergen**: HOMs from BBH mergers with ML (PCA + Ensemble NNs, TD) | `mlgw_NN.tex` |
| `2205.14066/` | **Thomas**: Accelerating precessing waveforms with ANNs (coprecessing frame, TD) | `main.tex` |
| `2411.14893/` | **Shi**: Rapid eccentric waveforms via deep learning (GPU-native, TD) | `templateArxiv.tex` |
| `2504.12420/` | **gwharmone**: First FD surrogate for eccentric harmonics (SVD + GPR) | `main.tex` |
| `2510.00116/` | **Chase Orbits**: Mean anomaly parameterization for eccentric surrogates | `main.tex` |

### Key Authors

- **CosmoPower**: A. Spurio Mancini, D. Piras, J. Alsing
- **Speculator**: Justin Alsing, Hiranya Peiris, Joel Leja
- **ripple**: Thomas D. P. Edwards, Kaze W. K. Wong, Kelvin K. H. Lam
- **IMRPhenomX**: Geraint Pratten, Cecilio García-Quirós, Marta Colleoni, Sascha Husa
- **Waveform Surrogates**: T. Grimbergen, L. Thomas, R. Shi, T. Islam (gwharmone), A. Maurya

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

## Literature Research

### Deep Research: Computational Paradigms in GW Modeling

**Source:** https://gemini.google.com/share/3500cf3dbd3a

**Document:** `docs/literature/deep-research/computational-paradigms-gw-modeling.md`

Comprehensive literature review of waveform surrogate techniques across time and frequency domains. Key findings:

- **Universal Triad**: Decomposition → Compression (SVD/PCA) → Regression (NN/GPR)
- **Validates our approach**: PCA + MLP, Mf as domain variable, aligned-spin first
- **Future techniques**: EIM for fast evaluation, GPR for uncertainty, Fourier feature embeddings

**Papers analyzed:** 2402.06587 (TD HOMs), 2205.14066 (TD precession), 2411.14893 (TD eccentric), 2504.12420 (FD gwharmone), 2510.00116 (mean anomaly)
