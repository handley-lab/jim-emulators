# Deep Research: Advanced NN Architectures for GW Emulation

Literature survey on neural network approaches beyond standard SVD+MLP.

## Survey Document

- `advanced-nn-architectures-survey.md` - Comprehensive review from Google Deep Research

## Appraisal

The deep research survey skewed toward fancier architectures (transformers, diffusion, HNNs) rather than the straightforward approaches that actually work well for this problem. Notably, it missed `2402.06587` (mlgw_NN) - a February 2024 paper using the exact same methodology we're pursuing (amp/phase → PCA → NN), achieving 10⁻⁴ mismatch with higher modes. This should have been a top hit for any survey on "neural network gravitational wave emulation."

## Papers by Category

### SVD-NN / PCA-NN Interpolants (Most Relevant)

| ArXiv ID | Title | Notes |
|----------|-------|-------|
| `2408.02470/` | obiwann: NN-based GW interpolant for low-latency | 4-layer MLP, 10⁻⁴ mismatch, ms generation |
| `2008.12932/` | ANN-Sur: Gravitational-wave surrogate models | SVD + NN interpolation |
| `2402.06587/` | **mlgw_NN**: ML model for time-domain GW with HM | Amp/phase → PCA → NN, SEOBNRv4HM, 10⁻⁴ mismatch |

### Latent Manifold / Autoencoders

| ArXiv ID | Title | Notes |
|----------|-------|-------|
| `2107.04312/` | Autoencoder-driven Spiral Representation Learning | Discovers spiral geometry in parameter space |
| `2101.06685/` | Conditional Autoencoders for GW generation | cAE achieving >97% overlap |

### Hamiltonian / Symplectic

| ArXiv ID | Title | Notes |
|----------|-------|-------|
| `2502.20881/` | HNN approach to fuzzball geodesics | Energy-conserving dynamics |
| `2102.12695/` | Learning orbital dynamics of BBH | Discovers PN corrections via UDEs |

### Fourier Neural Operators

| ArXiv ID | Title | Notes |
|----------|-------|-------|
| `2511.19364/` | Neural surrogates for GW detector design | FNO for interferometer optimization |

### Generative Models (Diffusion/Score-based)

| ArXiv ID | Title | Notes |
|----------|-------|-------|
| `2410.19956/` | SLIC: Score-Based Likelihood Characterization | Non-Gaussian noise modeling |
| `2208.05003/` | Wavelet Score-Based Generative Modeling | Multi-scale diffusion |
| `2402.15516/` | GLA-Grad: Griffin-Lim + Diffusion | Phase recovery for spectrograms |

### Time-Frequency / Transformers

| ArXiv ID | Title | Notes |
|----------|-------|-------|
| `2511.20731/` | Denoising GW in time-frequency domain | U-Net on spectrograms |
| `2512.02968/` | Dingo-T1: Transformers for GW PE | Flexible multi-detector handling |

## Training Data Sources

**Key finding**: Almost all papers train on semi-analytical approximants (SEOBNR, IMRPhenom, TaylorF2, EOB), not raw numerical relativity.

| Paper | Training Data |
|-------|---------------|
| `2408.02470` (obiwann) | SEOBNRv4_ROM (BBH), TaylorF2 (BNS) |
| `2008.12932` (ANN-Sur) | SEOBNRv4 |
| `2402.06587` (mlgw_NN) | SEOBNRv4HM (time domain, higher modes) |
| `2101.06685` (cAE) | EOB waveforms |
| `2107.04312` (Spiral) | SEOBNRv4, EOBNRv2 |
| `2102.12695` (UDE) | **NR trajectories** (SXS) - learns dynamics, not waveforms |
| `2410.19956` (SLIC) | IMRPhenomD (via ripple) |
| `2512.02968` (Dingo-T1) | **IMRPhenomXPHM** |

**Notable**: Dingo-T1 uses **IMRPhenomXPHM** - the same model we're targeting. Validates that XPHM is tractable for neural network approaches.

## Relevance to jim-emulators

**Directly applicable**:
- `2408.02470`, `2008.12932` - validate our SVD+MLP approach
- `2512.02968` - confirms IMRPhenomXPHM works with neural networks

**Background interest**: The rest solve different problems (detection, noise modeling, conservation laws) not central to our waveform emulation goal.
