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

---

## Session: 2024-12-22 (Continued - Parallel Work)

### 19. Review of Parameter Sensitivity Plots

Reviewed the sensitivity plots generated in the previous session. Key observations from discussion:

**Amplitude and Phase vs Real/Imaginary:**
- Looking at the plots, it's clear that **amplitude and phase** vary much more smoothly than real/imaginary parts
- Real/imaginary show rapid oscillations that would be difficult to emulate
- This confirms the CosmoPower/Speculator approach: emulate log-amplitude and unwrapped phase separately

**Higher-Order Mode Effects (XPHM):**
- In high-mass, high-spin cases, see dramatic amplitude reductions from higher-order modes
- "Wiggles" in the waveform come from precession effects (beating patterns)
- These are real physics, not numerical artifacts
- The "ratty" behavior at high frequencies is below the noise floor

**Time Domain Observations:**
- Geometric time (t/M) gives mass-independent waveforms
- Time domain shows much lower dynamic range than frequency domain
- Some waveforms look "crazy" in time domain - suggests frequency domain is the natural choice
- However, time domain might be useful for hybrid approaches where frequency domain looks ratty

**Mismatch Benchmarks (for context):**
- Waveform models vs numerical relativity: ~10⁻⁵ mismatch
- Machine precision (C vs Python floating point): ~10⁻¹⁶ mismatch
- Our target: 10⁻³ mismatch - well above model uncertainty

### 20. Future Ideas Captured

**Hybrid Time/Frequency Approach:**
- If certain parameter regions look "ratty" in frequency domain but smooth in time domain, could switch between representations
- Not for MVP, but worth exploring later

**Amplitude-Phase Regularization:**
- For time domain: h(t) = A(t) × exp(iφ(t))
- Neural net learning A(t) and φ(t) directly might be smoother than learning Re(h), Im(h)
- Would need regularization: φ should be monotonic, φ̇ ≠ 0
- Similar to WKB approximation concept

**Error Estimation:**
- CosmoPower doesn't provide uncertainty estimates
- Could add neural network uncertainty quantification later
- Mismatch itself provides good independent error estimate

### 21. XAS Comparison Plots

Started generating comparison plots with IMRPhenomXAS (aligned-spin only, no higher modes):
- XAS waveforms are much smoother (no HM wiggles)
- Useful sanity check: XPHM with aligned spins should give similar results to XAS
- Confirms higher-mode effects are the source of complex structure

### 22. Parallel Work Streams

Split into parallel tasks:
1. **Deep research** (Will): Survey existing GW emulation literature
2. **JAX/Flax preparation** (this session): Prepare documentation and snippets for neural network implementation

### 23. JAX/Flax Neural Network Preparation

Created comprehensive reference material for neural network implementation in `snippets/flax_nn.py`:

**Key Components Implemented:**

1. **SpeculatorActivation (Flax Module)**:
   - Implements Eq. 4 from Alsing et al. (2019): σ(x) = [γ + sigmoid(β·x)·(1-γ)]·x
   - Learnable α (sigmoid steepness) and γ (linear mixing) per neuron
   - Properties: smooth, infinitely differentiable (suitable for HMC)

2. **EmulatorMLP**:
   - 4 hidden layers × 512 units (CosmoPower default)
   - Input: normalized parameters (η, χ₁, χ₂)
   - Output: PCA coefficients (linear output layer)
   - Total parameters: ~820k

3. **Training Utilities**:
   - `create_train_state()`: Initialize with AdamW optimizer
   - `create_learning_rate_schedule()`: Warmup + cosine decay
   - `train_step()` / `eval_step()`: JIT-compiled training

4. **WaveformPCA Class**:
   - JAX-compatible PCA for waveform compression
   - Automatic component selection (99.99% variance)
   - `fit()`, `transform()`, `inverse_transform()` methods

5. **GWEmulator Class**:
   - Complete pipeline: params → NN → PCA → amplitude/phase
   - Ready for integration

**Verified:**
```
$ python snippets/flax_nn.py
Input shape: (1, 3)
Output shape: (1, 50)
Number of parameters: 819762
Speculator activation test:
  Input range: [-3.00, 3.00]
  Output range: [-0.28, 2.86]
```

**Dependencies Installed:**
- flax 0.12.2
- optax 0.2.6
- jax 0.8.2, jaxlib 0.8.2

---

### Next Steps

1. ☐ Create data generation script (LAL XPHM → HDF5 training data)
2. ☐ Implement PCA compression for amplitude/phase
3. ☐ Build neural network with Speculator activation (Flax)
4. ☐ Training pipeline with optax
5. ☐ Validation: mismatch < 10⁻³ target
6. ☐ Integration with ripple interface
7. ☐ XAS comparison plots (verify XPHM aligned-spin ≈ XAS)
8. ☐ Literature review: existing GW emulation approaches
