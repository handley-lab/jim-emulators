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

## Deep Research Literature

See `docs/literature/deep-research/` for survey of advanced NN architectures for GW emulation.

**Key papers for our approach:**

| ArXiv ID | Paper | Relevance |
|----------|-------|-----------|
| `2205.14066/` | **SEOBNN**: NN surrogate for SEOBNRv4PHM | **Key reference** - precessing + HM, co-precessing frame decomposition |
| `2402.06587/` | **mlgw_NN**: Time-domain GW with HM | Amp/phase → PCA → NN, 10⁻⁴ mismatch |
| `2408.02470/` | **obiwann**: SVD-NN interpolant | Validates SVD+MLP approach (10⁻⁴ mismatch) |
| `2512.02968/` | **Dingo-T1**: Transformers for GW PE | Uses IMRPhenomXPHM - validates our target model |

**Key background (2205.14066 SEOBNN)**: Closest existing work to our goal. Shows how to handle precession by decomposing into co-precessing frame modes + Euler angles. Achieves 100× speedup. Template for our IMRPhenomXPHM emulator.

**Training data insight**: All papers train on semi-analytical approximants (SEOBNR, IMRPhenom, EOB), not raw NR. This validates our approach of emulating IMRPhenomXPHM directly.
