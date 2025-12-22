# Handover Instructions

## Quick Start

```bash
# Clone the repo
git clone git@github.com:handley-lab/jim-emulators.git
cd jim-emulators

# Clone the reference codebases (not committed, needed for reference)
git clone git@github.com:alessiospuriomancini/cosmopower.git
git clone git@github.com:tedwards2412/ripple.git

# LALSimulation (sparse checkout - only the simulation package)
git clone --filter=blob:none --sparse https://git.ligo.org/lscsoft/lalsuite.git
cd lalsuite && git sparse-checkout set lalsimulation && cd ..
```

## What's Been Done

1. **Literature collected** - ArXiv paper sources in numbered directories (2106.03846/, etc.)
2. **Reference code cloned** - cosmopower, ripple, lalsimulation (gitignored)
3. **Documentation synthesized** from planning meeting transcripts
4. **Gemini consulted** for detailed implementation plan
5. **Strategy refined** - use XPHM directly, not XAS

## Key Files to Read

| File | Purpose |
|------|---------|
| `docs/project-overview.md` | Physical foundations, domain choices, problem spec |
| `docs/implementation-plan.md` | **START HERE** - Detailed step-by-step plan |
| `docs/gemini-plan.md` | Original Gemini output (for reference) |
| `docs/development-log.md` | Narrative of how we got here |
| `.claude/CLAUDE.md` | Context for Claude Code sessions |

## Current State

- **No code written yet** - just planning and documentation
- All infrastructure decisions made
- Ready to start implementation

## Next Steps (from implementation-plan.md)

1. **Repo Init**: Create `src/jim_emulators/` structure, requirements.txt
2. **Data Gen**: Write `scripts/generate_data.py` (LAL XPHM → HDF5)
3. **Components**: Implement PCA and neural network (Flax/JAX)
4. **Training**: Train on aligned spins (3D: η, χ₁z, χ₂z)
5. **Validation**: Mismatch tests against LAL
6. **Integration**: ripple-compatible interface

## Key Decisions Already Made

- **Waveform**: IMRPhenomXPHM (not XAS) - with aligned spins reduces to XHM
- **Domain**: Frequency domain, geometric units (Mf)
- **Architecture**: PCA + NN with Speculator activation (Flax/JAX)
- **Storage**: HDF5 for training data
- **Validation**: Mismatch < 10⁻³ target

## Development Log

Keep updating `docs/development-log.md` as you work - we're publishing the workflow alongside the emulators.

## Useful Commands

```bash
# Generate code2prompt for LLM context
code2prompt cosmopower --include "*.py" --output-file /tmp/cosmopower.md
code2prompt ripple --include "*.py" --output-file /tmp/ripple.md

# Example LAL waveform generation (see ripple/test/ for examples)
pip install lalsuite
```

## Reference: LAL Calling Convention

From ripple tests, a typical LAL call looks like:
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
    approximant       # e.g., lalsim.IMRPhenomXPHM
)
```

## Questions?

Check the transcripts (transcript-0921, transcript-0930, transcript-0944) for context on decisions made.
