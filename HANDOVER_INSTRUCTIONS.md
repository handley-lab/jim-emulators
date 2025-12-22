# Handover Instructions

## Quick Start

```bash
# Clone the repo
git clone git@github.com:handley-lab/jim-emulators.git
cd jim-emulators

# Switch to the active branch
git checkout theory-decomposition

# Clone the reference codebases (not committed, needed for reference)
git clone git@github.com:alessiospuriomancini/cosmopower.git
git clone git@github.com:tedwards2412/ripple.git

# LALSimulation (sparse checkout - only the simulation package)
git clone --filter=blob:none --sparse https://git.ligo.org/lscsoft/lalsuite.git
cd lalsuite && git sparse-checkout set lalsimulation && cd ..
```

## Why This Project Exists

Nobody else is doing frequency-domain, aligned-spin, higher-mode emulation with PCA + MLP. The literature focuses on harder problems (precession, eccentricity) because **IMRPhenomXHM is already fast** — it's an analytic model, not an ODE solver.

**Why we're doing it anyway:**
1. **JAX differentiability** — need gradients for jim's HMC sampler
2. **Faster than hand-porting** — testing emulation vs ripple's approach
3. **Proof of concept** — infrastructure extends to precessing/eccentric

## What's Been Done

1. **Literature collected** - 11 ArXiv papers in numbered directories
2. **Deep research** - comprehensive literature review (`docs/literature/deep-research/`)
3. **Theory documents** - two approved frameworks for FD and TD emulation
4. **Reference code cloned** - cosmopower, ripple, lalsimulation (gitignored)
5. **Documentation synthesized** from planning meetings and LLM reviews

## Key Files to Read

| File | Purpose |
|------|---------|
| `.claude/CLAUDE.md` | **START HERE** - Project context, paper index, theory summary |
| `docs/project-overview.md` | Physical foundations, domain choices, problem spec |
| `docs/implementation-plan.md` | Detailed step-by-step implementation plan |
| `docs/development-log.md` | Narrative of how we got here |
| `theory/frequency-domain-emulation.tex` | FD framework (15 pages, approved) |
| `theory/general-waveform-decomposition.tex` | TD framework (12 pages, approved) |

## Current State

- **Theory complete** - Two approved documents covering FD (immediate) and TD (future)
- **No code written yet** - Ready to start implementation
- **Branch**: `theory-decomposition` (ahead of `master`)

### Theory Documents

| Document | Pages | Domain | Target |
|----------|-------|--------|--------|
| `frequency-domain-emulation.tex` | 15 | Frequency | IMRPhenomXHM (aligned-spin, HOMs) |
| `general-waveform-decomposition.tex` | 12 | Time | Precessing/eccentric (future) |

### Papers in Repository

**Cosmological Emulators:** 2106.03846 (CosmoPower), 1911.11778 (Speculator)

**GW Waveforms:** 2302.05329 (ripple), 2306.17245 (ripple mismatch), 2004.06503 (XPHM), 2001.10914 (XHM)

**Waveform Surrogates:** 2402.06587 (Grimbergen HOMs), 2205.14066 (Thomas precessing), 2411.14893 (Shi eccentric), 2504.12420 (gwharmone), 2510.00116 (Chase Orbits)

## Next Steps

1. **Repo structure**: Create `src/jim_emulators/`, requirements.txt
2. **Data generation**: `scripts/generate_data.py` (LAL XHM → HDF5)
3. **PCA + MLP**: Implement in Flax/JAX following CosmoPower
4. **Training**: Aligned-spin parameter space (η, χ₁z, χ₂z)
5. **Validation**: Mismatch < 10⁻³ against LAL
6. **Integration**: ripple-compatible interface

## Key Decisions Made

- **Waveform**: IMRPhenomXHM (frequency domain, aligned-spin, higher modes)
- **Domain variable**: Mf (dimensionless frequency) — mass is not an intrinsic parameter
- **Architecture**: PCA + MLP with Speculator activation (Flax/JAX)
- **Representation**: Log-amplitude + unwrapped phase, log-spaced Mf grid
- **Storage**: HDF5 for training data
- **Validation**: Mismatch metric, target M < 10⁻³

## Development Log

Keep updating `docs/development-log.md` as you work — we're publishing the workflow alongside the emulators.

## Useful Commands

```bash
# Generate code2prompt for LLM context
code2prompt cosmopower --include "*.py" --output-file /tmp/cosmopower.md
code2prompt ripple --include "*.py" --output-file /tmp/ripple.md

# Download arxiv paper
python ~/.claude/skills/arxiv/scripts/arxiv.py <arxiv_id> --save

# Build theory PDFs
cd theory && latexmk -pdf *.tex
```

## Reference: LAL Calling Convention

From ripple tests:
```python
import lalsimulation as lalsim

hp, hc = lalsim.SimInspiralChooseFDWaveform(
    m1, m2,           # masses in kg
    s1x, s1y, s1z,    # spin 1 components
    s2x, s2y, s2z,    # spin 2 components
    distance,         # luminosity distance
    inclination,      # inclination angle
    phiRef,           # reference phase
    longAscNodes,     # longitude of ascending nodes
    eccentricity,     # eccentricity
    meanPerAno,       # mean anomaly
    deltaF,           # frequency step
    f_min, f_max,     # frequency range
    f_ref,            # reference frequency
    params,           # extra params (None usually)
    approximant       # e.g., lalsim.IMRPhenomXHM
)
```

## Questions?

- Check `docs/development-log.md` for decisions and rationale
- Check `theory/reviews/` for LLM review feedback
- Original transcripts: `transcript-0921`, `transcript-0930`, `transcript-0944`
