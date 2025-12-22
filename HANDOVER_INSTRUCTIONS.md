# Handover Instructions

## Quick Start

```bash
# Clone the repo
git clone git@github.com:handley-lab/jim-emulators.git
cd jim-emulators

# Clone the reference codebases (not committed, needed for reference)
git clone git@github.com:alessiospuriomancini/cosmopower.git
git clone git@github.com:GW-JAX-Team/ripple.git  # Note: GW-JAX-Team, not tedwards2412

# LALSimulation (sparse checkout - only the simulation package)
git clone --filter=blob:none --sparse https://git.ligo.org/lscsoft/lalsuite.git
cd lalsuite && git sparse-checkout set lalsimulation && cd ..

# Install dependencies
pip install -e ".[dev]"  # or just: pip install lalsuite jax matplotlib numpy
```

## What's Been Done

### Phase 1: Planning & Setup (Complete)
1. **Literature collected** - ArXiv paper sources in numbered directories
2. **Reference code cloned** - cosmopower, ripple, lalsimulation (gitignored)
3. **Documentation synthesized** from planning meeting transcripts
4. **Strategy decided** - use IMRPhenomXPHM directly, not XAS

### Phase 2: Simulation Infrastructure (Complete)
1. **LAL waveform wrapper** - Clean Python interface to LALSimulation
2. **Unit tests** - 14 tests covering waveform generation and match calculations
3. **Utility functions** - Frequency grids, match/mismatch, SNR, PSD loading
4. **Time domain utilities** - FFT to time domain in both physical and geometric units

### Phase 3: Parameter Sensitivity Analysis (Complete)
1. **54 sensitivity plots** generated showing waveform dependence on parameters
2. **Three representations**: physical frequency (Hz), geometric frequency (Mf), time domain (t and t/M)
3. **Three fiducial systems**: equal mass, unequal mass, high mass/high spin

## Key Files to Read

| File | Purpose |
|------|---------|
| `docs/implementation-plan.md` | **START HERE** - Detailed step-by-step plan |
| `docs/development-log.md` | Narrative of how we got here |
| `docs/project-overview.md` | Physical foundations, domain choices |
| `src/jim_emulators/waveforms/lal_waveforms.py` | LAL wrapper implementation |
| `src/jim_emulators/waveforms/utils.py` | Match calculations, time domain FFT |
| `scripts/parameter_sensitivity.py` | How sensitivity plots are generated |

## Current Codebase Structure

```
jim-emulators/
├── src/jim_emulators/
│   ├── __init__.py
│   └── waveforms/
│       ├── __init__.py
│       ├── lal_waveforms.py    # WaveformParameters, generate_fd_waveform()
│       └── utils.py            # compute_match(), fd_to_td(), geometric_fd_to_td()
├── tests/
│   └── test_lal_waveforms.py   # pytest tests (14 passing)
├── scripts/
│   └── parameter_sensitivity.py
├── figures/
│   └── sensitivity/            # 54 plots (frequency + time domain)
├── psds/
│   └── ET-D-psd.txt
├── docs/
│   ├── implementation-plan.md
│   ├── development-log.md
│   └── project-overview.md
├── pyproject.toml
└── (papers in numbered directories: 1911.11778/, 2004.06503/, etc.)
```

## Key Decisions Already Made

- **Waveform**: IMRPhenomXPHM (not XAS) - with aligned spins reduces to XHM
- **Domain**: Geometric units (Mf for frequency, t/M for time) for mass-independence
- **Architecture**: PCA + NN with Speculator activation (Flax/JAX)
- **Storage**: HDF5 for training data
- **Validation**: Mismatch < 10⁻³ target

## Running Tests

```bash
cd jim-emulators
pytest tests/ -v
```

## Regenerating Sensitivity Plots

```bash
python scripts/parameter_sensitivity.py
# Or for a single fiducial:
python scripts/parameter_sensitivity.py --fiducial equal_mass_nonspinning
```

## Next Steps (Priority Order)

### 1. Data Generation Script
Create `scripts/generate_data.py` to produce HDF5 training data:
- Sample parameter space using Latin Hypercube Sampling
- Generate waveforms on geometric frequency grid (Mf)
- Store amplitude and phase separately (or complex strain)
- Target: 10⁵ training samples, 10⁴ validation samples

```python
# Key parameters to sample:
# eta ∈ [0.05, 0.25]  (symmetric mass ratio)
# chi1z, chi2z ∈ [-0.99, 0.99]  (aligned spins)
# Mf grid: log-spaced, ~1000 points in [0.003, 0.25]
```

### 2. PCA Compression
Implement `src/jim_emulators/components/pca.py`:
- Fit PCA on training data (separately for amplitude and phase)
- Target ~99.99% explained variance
- Store basis vectors for reconstruction

### 3. Neural Network
Implement `src/jim_emulators/components/network.py`:
- Speculator activation function (Eq. 4 from 1911.11778)
- 4 hidden layers × 512 units
- Input: normalized (η, χ₁, χ₂)
- Output: PCA coefficients

### 4. Training Pipeline
Create `scripts/train.py`:
- MSE loss on PCA coefficients
- Adam optimizer with learning rate scheduling
- Early stopping on validation loss

### 5. Validation
Extend tests to compute mismatches against LAL across parameter space.

## Useful Code Snippets

### Generate a waveform:
```python
from jim_emulators.waveforms import WaveformParameters, generate_fd_waveform

params = WaveformParameters(
    mass_1=30.0, mass_2=25.0,
    chi1z=0.3, chi2z=-0.2,
    f_min=20.0, f_max=512.0, delta_f=0.125
)
freqs, hp, hc = generate_fd_waveform(params)
```

### Compute match between waveforms:
```python
from jim_emulators.waveforms import compute_match, load_psd
import jax.numpy as jnp

_, psd = load_psd("psds/ET-D-psd.txt", freqs)
match = compute_match(hp1, hp2, psd, freqs)
```

### Convert to time domain:
```python
from jim_emulators.waveforms import fd_to_td, geometric_fd_to_td

times, h_td = fd_to_td(hp, delta_f=0.125, center=True)
tM, h_tM = geometric_fd_to_td(hp_Mf, Mf_grid, center=True)
```

## Reference Papers (in repo)

| Directory | Paper | Key Content |
|-----------|-------|-------------|
| `2106.03846/` | CosmoPower | NN emulator architecture |
| `1911.11778/` | Speculator | Activation function (Eq. 4) |
| `2302.05329/` | ripple | JAX waveforms reference |
| `2004.06503/` | IMRPhenomXPHM | Waveform model details |

## Development Log

Keep updating `docs/development-log.md` as you work - we're publishing the workflow alongside the emulators.

## Questions?

Check the transcripts (`transcript-0921`, `transcript-0930`, `transcript-0944`) for context on decisions made.
