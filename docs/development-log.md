# Development Log: Jim Emulators

This document records the workflow used to bootstrap the jim-emulators project, demonstrating an AI-assisted approach to scientific software development.

## Session: 2024-12-22

### 1. Project Initialization

The project began with the goal of building JAX-based neural network emulators for gravitational wave waveforms, to be integrated into the jim ecosystem. The working directory was established at `/home/will/projects/jim-emulators/`.

### 2. Literature Collection

We systematically gathered reference papers from arXiv using targeted searches.

**Cosmological Emulators:**
- Searched for "cosmopower" and "speculator" papers
- Found the main CosmoPower paper via `ti:cosmopower` search
- Speculator required a more specific search: `ti:speculator+AND+ti:emulator`

**Gravitational Wave Waveforms:**
- Searched for IMRPhenomX papers via `abs:IMRPhenomXPHM` and author searches
- Found ripple via `ti:ripple+AND+ti:waveform`
- Located the ripple mismatch/recalibration paper via `abs:ripple+AND+abs:mismatch+AND+abs:waveform`

**Papers Downloaded:**

| ArXiv ID | Paper | Purpose |
|----------|-------|---------|
| 2106.03846 | CosmoPower | NN emulator architecture reference |
| 1911.11778 | Speculator | Activation function, PCA methodology |
| 2302.05329 | ripple | JAX waveforms, validation approach |
| 2306.17245 | ripple recalibration | Mismatch calculation methodology |
| 2004.06503 | IMRPhenomXPHM | Precessing higher modes waveform |
| 2001.10914 | IMRPhenomXHM | Non-precessing higher modes |

Each paper was downloaded with full LaTeX source using:
```bash
python ~/.claude/skills/arxiv/scripts/arxiv.py <arxiv_id> --save
```

### 3. Code Repository Collection

Three codebases were cloned for reference:

**CosmoPower** (SSH):
```bash
git clone git@github.com:alessiospuriomancini/cosmopower.git
```
The neural network emulator we're adapting. Key files: Python implementation of PCA + NN architecture.

**ripple** (SSH):
```bash
git clone git@github.com:tedwards2412/ripple.git
```
JAX-based GW waveforms with mismatch calculations. Contains LAL calling conventions and validation infrastructure.

**LALSimulation** (HTTPS, sparse checkout):
```bash
git clone --filter=blob:none --sparse https://git.ligo.org/lscsoft/lalsuite.git
cd lalsuite && git sparse-checkout set lalsimulation
```
Reference C implementations of IMRPhenomX waveforms. Used HTTPS as SSH access to LIGO GitLab was not available.

These repositories are gitignored (not committed) as they are large external dependencies.

### 4. Meeting Transcript Processing

Two planning meeting transcripts were processed:
- `transcript-0921`: Initial planning discussion
- `transcript-0930`: Extended discussion with problem specification

Key insights extracted from transcripts:

**Physical Foundations:**
- Waveforms depend on `Mf` (mass × frequency), not `f` alone
- Core parameter space is 7D maximum: η + 6 spin components
- Aligned-spin case reduces to 3D: η, χ₁ᵤ, χ₂ᵤ
- Extrinsic parameters (distance, phase, inclination) handled analytically

**Problem Specification:**
- Amplitude varies over ~5 orders of magnitude
- Phase varies by ~100-200 radians
- Frequency dynamic range ~10³
- Target mismatch accuracy: ~10⁻³

**Development Strategy:**
- Store training data as complex strain (flexible for amp/phase or real/imag)
- Install LAL via `pip install lalsuite`
- Use ripple examples for LAL calling conventions
- Consider external ML literature review once benchmarks established

### 5. Documentation Synthesis

The transcript insights were synthesized into structured documentation:
- `docs/project-overview.md`: Physical foundations, domain choices, development phases
- `.claude/CLAUDE.md`: Project-specific notes for AI assistant context

### 6. Version Control Setup

The project was initialized as a git repository with:
- `.gitignore` excluding cloned repositories (cosmopower/, ripple/, lalsuite/)
- Papers committed (LaTeX sources are small, valuable reference)
- Documentation committed

Repository created on GitHub:
```bash
gh repo create handley-lab/jim-emulators --public --source . --push
```

**Repository:** https://github.com/handley-lab/jim-emulators

### 7. Codebase Analysis

Used `code2prompt` to generate token-counted summaries for LLM analysis:

| Source | Tokens | Command |
|--------|--------|---------|
| Main .tex papers | 174K | `--include "*/ms.tex" --include "*/CosmoPower.tex" ...` |
| cosmopower (*.py) | 29K | `--include "*.py"` |
| ripple (*.py) | 120K | `--include "*.py"` |
| lalsim IMRPhenomX | 718K | `--include "*IMRPhenomX*.c"` (excluded from analysis) |

LALSimulation was excluded from LLM analysis due to size (6M+ tokens for all C/H files).

### 8. Implementation Planning

The combined codebase (papers + cosmopower + ripple, ~320K tokens) was sent to Gemini for detailed implementation planning:

```bash
code2prompt ... --output-file /tmp/jim_emulators_full.md
code2prompt cosmopower --include "*.py" --output-file /tmp/cosmopower_py.md
code2prompt ripple --include "*.py" --output-file /tmp/ripple_py.md
```

Gemini produced `docs/gemini-plan.md` containing:
1. Architecture analysis (PCA + NN with Speculator activation)
2. Data generation pipeline (LAL → HDF5)
3. Step-by-step implementation plan
4. Training pipeline specification
5. Validation methodology
6. Integration strategy

### 9. Current State

**Committed Files:**
```
jim-emulators/
├── .claude/CLAUDE.md          # AI assistant context
├── .gitignore                  # Excludes cloned repos
├── docs/
│   ├── project-overview.md    # Synthesized requirements
│   ├── gemini-plan.md         # Detailed implementation plan
│   └── development-log.md     # This file
├── transcript-0921            # Planning meeting 1
├── transcript-0930            # Planning meeting 2
├── 1911.11778/                # Speculator paper
├── 2001.10914/                # IMRPhenomXHM paper
├── 2004.06503/                # IMRPhenomXPHM paper
├── 2106.03846/                # CosmoPower paper
├── 2302.05329/                # ripple paper
└── 2306.17245/                # ripple mismatch paper
```

**Cloned (not committed):**
```
├── cosmopower/                # NN emulator reference
├── ripple/                    # JAX waveforms reference
└── lalsuite/lalsimulation/    # LAL reference (sparse)
```

### Tools Used

- **Claude Code**: Primary development assistant
- **arxiv skill**: Paper search and download
- **code2prompt**: Codebase summarization for LLM context
- **mcp__llm-chat__ask**: Gemini consultation for planning
- **gh CLI**: GitHub repository management

### 10. Strategy Refinement

A third transcript (`transcript-0944`) captured further strategic decisions:

**Waveform Model Decision:**
- Use **IMRPhenomXPHM directly**, not XAS or simpler waveforms
- XPHM with aligned spins reduces to XHM automatically
- This builds the full infrastructure once; simplify by restricting parameter space, not changing model
- Higher modes produce high-frequency "wiggles" - important for fidelity even if below noise floor

**Grid Strategy Confirmed:**
- Log-spaced grid in Mf (geometric frequency)
- Parameter space: (η, χ₁ᵤ, χ₂ᵤ) for aligned-spin case

**Publication Notes:**
- Save Claude Code conversation exports for workflow documentation
- Development log should be narrative, capturing the story of the build

### 11. Handover Prepared

Created `HANDOVER_INSTRUCTIONS.md` for continuation on a different machine:
- Quick start commands (clone repo + reference codebases)
- Key files to read
- Current state summary
- Next steps checklist
- Key decisions already made
- LAL calling convention reference

---

## Session: 2024-12-22 (Continued - Machine Handover)

### 12. Machine Handover & Environment Setup

Project continued on a new machine. Following the handover instructions:

```bash
# Cloned reference repositories
git clone git@github.com:alessiospuriomancini/cosmopower.git
git clone git@github.com:GW-JAX-Team/ripple.git  # Note: correct repo is GW-JAX-Team, not tedwards2412
git clone --filter=blob:none --sparse https://git.ligo.org/lscsoft/lalsuite.git
cd lalsuite && git sparse-checkout set lalsimulation
```

Verified LAL installation: `lalsimulation.__version__ = 6.2.0`

### 13. LAL Waveform Wrapper Implementation

Created a clean, testable LAL simulation wrapper with proper unit handling.

**Package Structure:**
```
src/jim_emulators/
├── __init__.py
└── waveforms/
    ├── __init__.py
    ├── lal_waveforms.py    # LAL wrapper with WaveformParameters dataclass
    └── utils.py            # Frequency grids, match calculations, PSD loading
```

**Key Design Decisions:**

1. **WaveformParameters Dataclass**: Clean interface encapsulating all waveform parameters
   - Automatic mass/spin swapping to ensure m1 >= m2
   - Derived quantities (chirp mass, eta, chi_eff) as properties
   - Input in astrophysical units (solar masses, Mpc), internal conversion to SI

2. **Supported Approximants**: IMRPhenomXPHM, IMRPhenomXHM, IMRPhenomXAS, IMRPhenomD

3. **JAX-Compatible Utilities**:
   - `noise_weighted_inner_product()`: For match calculations
   - `compute_match()`, `compute_mismatch()`: Waveform comparison
   - `get_geometric_frequency_grid()`: Log-spaced Mf grid for emulation
   - `physical_to_geometric_frequency()`: Unit conversion

**Reference Sources:**
- `ripple/tests/old_tests/test_IMRPhenomX.py`: LAL calling conventions
- `ripple/tests/benchmark_waveform.py`: Match calculation patterns
- `bilby/gw/utils.py:663-705`: Clean wrapper design

### 14. Test Suite

Created comprehensive test suite (`tests/test_lal_waveforms.py`):

```bash
pytest tests/test_lal_waveforms.py -v
# Result: 14 passed in 7.25s
```

**Tests Cover:**
- Parameter dataclass creation and validation
- Mass/spin swapping
- Derived quantities (eta, chirp mass, chi_eff)
- Waveform generation for all approximants
- Distance scaling (amplitude ∝ 1/D_L)
- Amplitude/phase extraction
- Self-match = 1.0
- Match calculation consistency

### 15. Parameter Sensitivity Analysis

Created `scripts/parameter_sensitivity.py` to visualize waveform dependence on parameters.

**Fiducial Parameter Sets:**
1. `equal_mass_nonspinning`: m1=m2=30 M☉, chi1z=chi2z=0
2. `unequal_mass_aligned_spin`: m1=35, m2=25, chi1z=0.3, chi2z=-0.2
3. `high_mass_high_spin`: m1=60, m2=40, chi1z=0.8, chi2z=0.6

**Parameters Varied:**
- Primary mass m1
- Secondary mass m2
- Primary spin chi1z
- Secondary spin chi2z
- Inclination iota
- Symmetric mass ratio eta (at fixed total mass)

**Output:**
Generated 36 figures in `figures/sensitivity/`:
- 2x2 grids showing: log amplitude, unwrapped phase, real part, imaginary part
- Both physical frequency (Hz) and geometric frequency (Mf) representations
- Color gradient indicates parameter variation

**Key Observations:**
- Mass changes shift merger frequency (lower mass = higher frequency)
- Spin affects both amplitude and phase evolution
- Inclination primarily affects amplitude through antenna pattern
- Geometric frequency Mf provides mass-independent representation (crucial for emulation)

### 16. Project Configuration

Created `pyproject.toml` with:
- Package metadata
- Dependencies: numpy, jax, jaxlib, lalsuite, matplotlib, h5py, scipy
- Optional dev dependencies: pytest, black, isort, flake8
- Optional training dependencies: flax, optax, tqdm

### 17. Time Domain Utilities

Added frequency-to-time-domain conversion utilities to enable visualization and analysis in both physical and geometric time domains.

**New functions in `utils.py`:**
- `fd_to_td()` - Frequency to time domain with automatic centering at merger
- `fd_to_td_centered_at_merger()` - Uses phase shift property for exact positioning
- `geometric_fd_to_td()` - Transforms h(Mf) → h(t/M) for mass-independent representation
- `geometric_time_to_physical()` / `physical_time_to_geometric()` - Unit conversions
- `interpolate_to_uniform_grid()` - For non-uniform grids (e.g., log-spaced Mf) before FFT

**Key insight:** Transforming from geometric frequency Mf to geometric time t/M gives a **mass-independent time-domain waveform**. All systems with the same intrinsic parameters (η, χ₁, χ₂) produce identical waveforms in t/M coordinates. Physical time is then simply:
```
t_seconds = (t/M) × M_total × 4.926×10⁻⁶ s
```

### 18. Extended Parameter Sensitivity Plots

Extended the sensitivity analysis to include time-domain visualizations:

**Plot types (54 total = 3 fiducials × 6 parameters × 3 types):**
1. `sensitivity_*.png` - Physical frequency domain (4 panels: log amplitude, phase, real, imaginary)
2. `sensitivity_Mf_*.png` - Geometric frequency domain (4 panels: log amplitude, phase, real, imaginary)
3. `sensitivity_td_*.png` - Time domain (4 panels: h(t), |h(t)|, h(t/M), |h(t/M)|)

Increased grid resolution from 15 to 30 parameter samples for smoother color gradients.

**Observations from time-domain plots:**
- Waveforms in geometric time t/M are mass-independent (only η and spins matter)
- Higher mass systems have longer duration in physical time but identical shape in t/M
- Spin affects the ringdown oscillation frequency visible in the envelope
- Inclination changes amplitude but not the intrinsic waveform shape

---

### Current State

**Implemented:**
```
jim-emulators/
├── src/jim_emulators/
│   ├── __init__.py
│   └── waveforms/
│       ├── __init__.py
│       ├── lal_waveforms.py    # LAL wrapper, WaveformParameters
│       └── utils.py            # Match, PSD, time domain utilities
├── tests/
│   └── test_lal_waveforms.py   # 14 tests passing
├── scripts/
│   └── parameter_sensitivity.py
├── figures/
│   └── sensitivity/            # 54 parameter sensitivity plots
├── psds/
│   └── ET-D-psd.txt            # Einstein Telescope PSD
├── pyproject.toml
└── (documentation + papers)
```

**Validated:**
- LAL waveform generation works correctly (all 4 approximants)
- All 14 unit tests pass
- Parameter sensitivity plots generated in frequency and time domains
- Time domain centering works correctly

---

## Session: 2024-12-22 (Continued - Mode Selection Investigation)

### 19. Unified Sensitivity Plots

Refactored parameter sensitivity plots to show all representations in a single figure:
- 2×3 grid: Amplitude, Phase, Time Domain / Real, Imaginary, Envelope
- Both physical units (Hz, ms) and geometric units (Mf, t/M) versions
- Fiducial parameter values displayed in title

### 20. XAS vs XPHM Comparison

Generated parallel sensitivity plots for IMRPhenomXAS to compare against XPHM:
- XAS: Aligned-spin, dominant (2,2) mode only
- XPHM: Higher modes + precession
- Same fiducial parameters for direct comparison
- Output: `figures/sensitivity/` (XPHM) and `figures/sensitivity_xas/` (XAS)

**Key observations:**
- Higher modes visible as "wiggles" in amplitude at high frequencies
- Most pronounced for unequal mass ratios and edge-on inclinations

### 21. Mode Selection Implementation

Added mode selection functionality to the LAL wrapper:

```python
from jim_emulators.waveforms import generate_fd_waveform, WaveformParameters

params = WaveformParameters(mass_1=35.0, mass_2=25.0, ...)

# Generate with all modes (default)
freqs, hp, hc = generate_fd_waveform(params)

# Generate with only (2,±2) mode
freqs, hp, hc = generate_fd_waveform(params, mode_array=[(2, 2), (2, -2)])
```

**New functions:**
- `generate_fd_waveform(params, mode_array=None)` - now accepts mode selection
- `create_mode_array(modes)` - creates LAL ModeArray from list of (l, m) tuples
- `get_available_modes(approximant)` - returns available modes for an approximant

### 22. Mode Additivity Test

**Question:** Is h(f) = Σ_{lm} h_lm(f) for XPHM?

**Test:** Generated waveforms with individual mode groups and compared sum to full waveform.

**Results:**
```
Max |h_full - sum(h_lm)|:    1.4e-31 (machine precision)
Match(h_full, sum(h_lm)):    1.000000000000000
```

**Conclusion: ✓ Modes ARE additive** (to machine precision)

### 23. XPHM(2,2) vs XAS Comparison

**Question:** Does XPHM with only (2,±2) modes equal XAS?

**Test:** Generated both and computed match.

**Results:**
| Comparison | Match | Mismatch |
|------------|-------|----------|
| XAS vs XPHM (all modes) | 0.9966 | 3.4×10⁻³ |
| XAS vs XPHM (2,±2 only) | 0.9983 | 1.7×10⁻³ |

**Conclusion:** XPHM(2,2) is closer to XAS than XPHM(all), but **not identical** (mismatch ~1.7×10⁻³). This is because XAS and XPHM are separately calibrated models, not the same code with mode selection.

### 24. Important LAL Convention Discovery

**Finding:** When requesting individual modes in LAL:
- Positive m modes (2,2), (3,3), etc. include BOTH +m and -m contributions
- Negative m modes (2,-2), (3,-3), etc. return ZERO

This means for mode selection, use only positive m values:
```python
mode_array = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]  # Correct
# NOT: [(2, 2), (2, -2), (2, 1), (2, -1), ...]  # Redundant
```

### 25. Mode-by-Mode Sensitivity Plots

Generated sensitivity plots for each individual spherical harmonic mode to understand if amplitude/phase are smoother when viewed per-mode.

**Output structure:**
```
figures/sensitivity_modes/
├── mode_22/   # (2,±2) dominant quadrupole - 36 plots
├── mode_21/   # (2,±1) subdominant - 34 plots
├── mode_33/   # (3,±3) octupole - 34 plots
├── mode_32/   # (3,±2) mixed - 36 plots
└── mode_44/   # (4,±4) hexadecapole - 36 plots
```

**Key Finding: Individual modes are MUCH smoother than the full waveform!**

| Mode | Amplitude | Phase | Notes |
|------|-----------|-------|-------|
| (2,2) | Smooth, monotonic | Smooth | Dominant, easy to emulate |
| (2,1) | Smooth | Smooth | Subdominant, vanishes at ι=0 |
| (3,3) | Smooth | Smooth | Higher frequency oscillations in time |
| (3,2) | Smooth | Smooth | Mixed mode |
| (4,4) | Smooth | Smooth | Highest frequency content |
| **Full** | **Wiggly** | Complex | Mode interference creates oscillations |

**Implication for emulation:** The wiggles in the full waveform come from **mode interference**, not intrinsic complexity. Emulating each mode separately then summing (since modes ARE additive) may be much easier than emulating the full waveform directly.

### 26. LAL Default Modes and Numerical Settings

**Default modes for IMRPhenomXPHM/XHM** (from `LALSimIMRPhenomXHM.c:142-151`):
```
(2, ±2)  - Dominant quadrupole (from XAS calibration)
(2, ±1)  - Subdominant quadrupole
(3, ±3)  - Octupole
(3, ±2)  - Mixed
(4, ±4)  - Hexadecapole
```

**Multibanding (numerical optimization):**

LAL uses "multibanding" to speed up waveform generation by computing on a coarser frequency grid and interpolating. This can introduce numerical artifacts, especially at low amplitudes.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PhenomXHMThresholdMband` | 10⁻³ | Threshold for XHM multibanding |
| `PhenomXPHMThresholdMband` | varies | Threshold for XPHM Euler angles |

**To disable multibanding (for maximum precision):**
```python
import lal
import lalsimulation as lalsim

laldict = lal.CreateDict()
lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(laldict, 0.0)
lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(laldict, 0.0)
# Pass laldict to SimInspiralChooseFDWaveform
```

Setting threshold to 0 disables multibanding entirely, giving exact (but slower) evaluation. This may improve smoothness at low amplitudes.

**Other numerical thresholds found in LAL:**
- Ringdown denominator floor: `1e-16` (prevents division by zero)
- Amplitude thresholds: `0.1/ampNorm` (mode 32), `0.01/ampNorm` (others)
- Mass ratio validation: `1e-12` tolerance for boundary checks

### 27. Multibanding Test and Precision Mode Generation

**Test: Effect of disabling multibanding** (`scripts/test_multibanding.py`)

| Metric | Value |
|--------|-------|
| Match (h_multibanding, h_exact) | 0.999999930627155 |
| Mismatch | 6.9×10⁻⁸ |
| Max phase difference | ~1.5 rad (at high frequency) |
| Max relative amplitude diff | ~10⁻⁶ |

**Conclusion:** Multibanding is NOT the source of the amplitude "wiggles" in the full waveform. The wiggles are real physics from mode interference, not numerical artifacts. Disabling multibanding improves precision but doesn't fundamentally change smoothness.

**Generated mode-by-mode plots with BOTH multibanding settings:**

Updated `scripts/parameter_sensitivity_by_mode.py` to support command-line control:
```bash
# With multibanding (LAL default, faster):
python scripts/parameter_sensitivity_by_mode.py --multibanding

# Without multibanding (maximum precision):
python scripts/parameter_sensitivity_by_mode.py
```

Generated 352 total plots (176 per setting) for side-by-side comparison:
```
figures/
├── sensitivity_modes_multibanding/   # 176 plots (LAL default)
│   ├── mode_22/  # 36 plots
│   ├── mode_21/  # 34 plots
│   ├── mode_33/  # 34 plots
│   ├── mode_32/  # 36 plots
│   └── mode_44/  # 36 plots
│
└── sensitivity_modes_precise/        # 176 plots (multibanding disabled)
    ├── mode_22/  # 36 plots
    ├── mode_21/  # 34 plots
    ├── mode_33/  # 34 plots
    ├── mode_32/  # 36 plots
    └── mode_44/  # 36 plots
```

**Key insight confirmed:** Individual modes remain smooth with both settings. The difference between multibanding ON/OFF is minimal (mismatch ~7×10⁻⁸). Smoothness is intrinsic to the physics of each mode, not an artifact of numerical interpolation.

---

### Current State

**New files added:**
```
scripts/
├── parameter_sensitivity_xas.py      # XAS sensitivity plots
├── parameter_sensitivity_by_mode.py  # Mode-by-mode analysis (supports --multibanding flag)
├── test_mode_selection.py            # XPHM(2,2) vs XAS test
├── test_mode_additivity.py           # Mode additivity verification
└── test_multibanding.py              # Multibanding effect comparison

figures/
├── sensitivity/                      # XPHM full waveform plots (36)
├── sensitivity_xas/                  # XAS plots (36)
├── sensitivity_modes_multibanding/   # Per-mode with multibanding (176)
├── sensitivity_modes_precise/        # Per-mode without multibanding (176)
├── mode_selection_test.png
├── mode_additivity_test.png
├── multibanding_comparison.png
├── multibanding_modes_comparison.png
└── multibanding_smoothness_comparison.png
```

**Updated files:**
- `src/jim_emulators/waveforms/lal_waveforms.py` - added mode selection + multibanding control
- `src/jim_emulators/waveforms/__init__.py` - exported new functions

---

### Next Steps

1. ☑ Add multibanding control to wrapper (done)
2. ☐ Create data generation script (LAL XPHM → HDF5 training data)
3. ☐ Implement PCA compression for amplitude/phase (per-mode)
4. ☐ Build neural network with Speculator activation (Flax)
5. ☐ Training pipeline with optax
6. ☐ Validation: mismatch < 10⁻³ target
7. ☐ Integration with ripple interface
