# Development Log: Jim Emulators

This document records the workflow used to bootstrap the jim-emulators project, demonstrating an AI-assisted approach to scientific software development.

## Session: 2024-12-22 (Morning - Gail's, Cambridge)

*Will Handley and James Alvey met at Gail's bakery in Cambridge. We had intended to work from Gonville & Caius College (both being Fellows), but fire damage from the previous day closed the college. The session ran until 10am when Will had to attend College Council and James had calls with prospective PhD students.*

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

Two planning meeting transcripts were processed (now in `transcripts/`):
- `2025-12-22-09-07-00.txt`: Initial planning discussion
- `2025-12-22-12-10-00.txt`: Extended discussion with problem specification

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
├── transcripts/
│   ├── 2025-12-22-09-07-00.txt  # Initial planning meeting
│   └── 2025-12-22-12-10-00.txt  # Progress review meeting
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

The initial planning transcript (`transcripts/2025-12-22-09-07-00.txt`) captured strategic decisions:

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

## Session: 2024-12-22 (Late Morning - James solo)

*While Will attended College Council, James continued development, building out the LAL wrapper infrastructure and parameter sensitivity analysis.*

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

## Session: 2024-12-22 (Afternoon - KICC K19, Cambridge)

*Reconvened at Will's office in the Kavli Institute for Cosmology (KICC), room K19. Reviewed James's progress from the morning and began parallel work streams: Will on deep research literature survey, James on mode selection and sensitivity analysis.*

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

### 24. Training Pipeline Verification

Created `scripts/train_toy_example.py` to verify the full training pipeline works before using real waveform data.

**Toy Problem:**
- 3D input → 50D output (mimicking (η, χ₁, χ₂) → PCA coefficients)
- Smooth nonlinear target function (sinusoids + polynomials)
- 10k training samples, 1k validation samples

**Results:**
```
Initial loss:  0.464
Final loss:    0.0002
Improvement:   2370x
Training time: ~11 seconds (96 epochs)
Speed:         ~9 epochs/second on CPU
```

**Verified:**
- SpeculatorActivation module works correctly
- EmulatorMLP with 4×256 hidden layers trains successfully
- AdamW optimizer converges smoothly
- Early stopping triggers appropriately
- JIT compilation works
- Gradients are computable (critical for HMC inference)

The JAX/Flax/Optax training infrastructure is ready for real waveform data.

### 25. Training Visualization

Added visualization to the toy training example (`figures/training/`):

1. **toy_training_results.png**: Training curves, prediction scatter, residual histogram, per-dimension MSE
2. **toy_predictions_overlay.png**: 6 random samples showing ground truth (blue) vs emulation (red dashed) - nearly perfect overlap
3. **toy_parameter_variation.png**: How output "spectrum" varies with each input parameter, comparing truth (solid) vs emulation (dashed) across parameter range

These plots demonstrate the network learns smooth interpolation across the parameter space - exactly what's needed for waveform emulation.

**Implementation Note:** The Flax/JAX implementation was done from existing knowledge of the framework (no external lookup needed). The Speculator activation formula was extracted from CosmoPower's TensorFlow code and translated to Flax Linen API.

### 26. External Code Review (GPT-5 + Gemini)

Submitted the Flax code to GPT-5 and Gemini (with grounding) for critical review. Key findings synthesized in `docs/jax-flax-best-practices.md`:

**High Priority Issues:**
1. **Unconstrained γ parameter**: Should use sigmoid on logits to enforce γ ∈ (0,1) per the Speculator paper
2. **Normalization not checkpointed**: Store as Flax variables, not Python attributes

**Medium Priority:**
3. **Python batch loop inefficient**: Replace with `lax.scan` for jitted epoch
4. **No gradient clipping**: Add `optax.clip_by_global_norm(1.0)` for stability
5. **No checkpointing**: Add Orbax for model saving

**Architecture Suggestions:**
- Use shape inference instead of explicit `features` argument
- Consider LayerNorm between layers
- Separate heads for amplitude/phase (different scales)
- Use sklearn for PCA fitting, JAX only for transform (numerical stability)
- Inject PCA basis as constant for end-to-end differentiability (needed for HMC)

**Performance:**
- Training in float32, inference in float64 for speed/accuracy tradeoff
- Buffer donation with `donate_argnums=(0,)`

These improvements will be incorporated when we move to real waveform training.

### 27. Implementing Review Recommendations

Applied fixes based on GPT-5 and Gemini reviews:

**Changes Made:**
1. **Constrained γ to (0,1)** via sigmoid on learnable `gamma_logits`
2. **Added gradient clipping** with `optax.clip_by_global_norm(1.0)`
3. **Shape inference** - removed explicit `features` argument
4. **Removed `jax_enable_x64`** from library module (should be set only at process start)
5. **Fixed `__main__` bug** - removed obsolete `features=100` argument

**Empirical Test: Gamma Initialization**

Both reviewers suggested starting more linear (+2.0 logits) would improve convergence. We tested both:

| Init | sigmoid(logits) | Behavior | Improvement | Final Val Loss |
|------|-----------------|----------|-------------|----------------|
| -2.0 | ≈ 0.12 | Mostly gated | 1330x | 0.000349 |
| +2.0 | ≈ 0.88 | Mostly linear | 176x | 0.003791 |

**Surprising Result:** The mostly-gated initialization (-2.0) significantly outperformed the mostly-linear start, contrary to theoretical expectations. For our toy task with nonlinear mappings, the gated activation helps the network learn complex transformations more effectively.

**Conclusion:** Keep -2.0 initialization. The theory ("linear is easier to optimize") may apply to simpler tasks, but for scientific emulation with nonlinear structure, allowing the network to start gated is beneficial.

**Updated Documentation:**
- `docs/jax-flax-best-practices.md` updated with empirical findings
- Added initialization comparison data

## Session: 2024-12-22 (Parallel Branch: wh-session-1251)

### 28. Deep Research Literature Review

Received comprehensive literature survey on advanced NN architectures for GW emulation from Google Deep Research. Organized into `docs/literature/deep-research/`.

**First Survey:** [Gemini Deep Research](https://gemini.google.com/share/ce2032c993f4)

**Papers Downloaded (12 total):**

| Category | ArXiv IDs |
|----------|-----------|
| SVD-NN interpolants | `2408.02470` (obiwann), `2008.12932` (ANN-Sur) |
| Latent manifolds | `2107.04312` (spiral), `2101.06685` (cAE) |
| Hamiltonian/symplectic | `2502.20881` (HNN), `2102.12695` (UDE dynamics) |
| FNO | `2511.19364` (detector design) |
| Generative models | `2410.19956` (SLIC), `2208.05003` (wavelet), `2402.15516` (GLA-Grad) |
| Time-frequency/transformers | `2511.20731` (denoising), `2512.02968` (Dingo-T1) |

**Key Finding - Training Data Sources:**

Almost all papers train on semi-analytical approximants, not raw NR:

| Paper | Training Data |
|-------|---------------|
| obiwann | SEOBNRv4_ROM, TaylorF2 |
| ANN-Sur | SEOBNRv4 |
| cAE | EOB |
| Spiral | SEOBNRv4, EOBNRv2 |
| UDE | NR trajectories (exception - learns dynamics) |
| SLIC | IMRPhenomD |
| Dingo-T1 | **IMRPhenomXPHM** |

**Important Validation:** Dingo-T1 uses IMRPhenomXPHM - the same model we're targeting. Confirms XPHM is tractable for neural network approaches.

**obiwann paper details (2408.02470):**
- Architecture: 4-layer MLP × 512 neurons, ReLU activations
- Input: 4D intrinsic parameters (m₁, m₂, s₁z, s₂z)
- Output: SVD coefficients for waveform reconstruction
- Performance: 10⁻⁴ mismatch (BBH), 10⁻⁵ mismatch (BNS)
- Speed: 0.28 ms/waveform (GPU), 0.84 ms for 10⁴ batch
- Training: 10⁵ samples, ~10 min on GPU

This validates our CosmoPower/Speculator approach - simple SVD+MLP achieves state-of-the-art results.

**Appraisal of Deep Research:**

The first survey skewed toward fancier architectures (transformers, diffusion, HNNs) rather than the straightforward approaches that actually work well for this problem. Notably, it missed `2402.06587` (mlgw_NN) - a February 2024 paper using the exact same methodology we're pursuing (amp/phase → PCA → NN), achieving 10⁻⁴ mismatch with higher modes.

**Second Survey (seeded with 2402.06587):** [Gemini Deep Research](https://gemini.google.com/share/b4a663061349)

A follow-up deep research seeded with the mlgw_NN paper found the actual PCA+NN lineage:

| Paper | Contribution |
|-------|-------------|
| mlgw (Schmidt et al.) | Foundational work, first robust Python package |
| mlgw_bns | Extension to binary neutron stars with tidal effects |
| NRSurNN3dq4 (2412.06946) | Transfer learning: pre-train on approximants, fine-tune on NR |
| SEOBNRE_AI (2411.14893) | Eccentric binaries via adaptive resampling |
| Transformer HOM (2409.03833) | Sequence-to-sequence, better out-of-distribution generalization |

Key insight from second survey: The PCA+NN architecture is a "stable design pattern" that has been validated across multiple groups (Utrecht, Pisa, SXS). The "manifold hypothesis" for GW - that valid waveforms lie on a low-dimensional manifold - is validated by PCA's success.

**Key Background Paper (missing from both surveys):**

`2205.14066` (SEOBNN) - Neural network surrogate for **SEOBNRv4PHM** (precessing + higher modes). This is the closest existing work to our IMRPhenomXPHM target:

- Decomposes precessing waveform into co-precessing frame modes (8 components) + Euler angles (3 components)
- Uses reduced basis + empirical interpolation + neural networks
- Achieves 100× speedup on CPU (18ms per waveform from 20Hz for 44 M☉)
- GPU batching provides further acceleration
- Reduces inference timescale from weeks to hours

This paper provides the template for handling precession in our emulator.

**Third Survey (seeded with 2205.14066):** [Gemini Deep Research](https://gemini.google.com/share/9a2971557222)

**Fourth Survey (SEOBNN deep-dive):** [Gemini Deep Research](https://gemini.google.com/share/59343e3a7d1b)

The best survey - detailed technical breakdown of the coprecessing frame technique:

1. **Coprecessing frame**: Transform to non-inertial frame tracking orbital plane → waveform simplifies to resemble aligned-spin case
2. **Decomposition**: Model separately - coprecessing modes (smooth, easy for NN) + Euler angles α(t), β(t), γ(t) (precession dynamics)
3. **SVD basis**: Dimensionality reduction ensures output "looks like a gravitational wave"
4. **Empirical interpolation**: Predict waveform at specific time nodes rather than abstract coefficients → better generalization

**Performance benchmarks (SEOBNN):**

| Hardware | Mode | Time/waveform | Speedup vs SEOBNRv4PHM |
|----------|------|---------------|------------------------|
| CPU | Serial | 18 ms | 100-200× |
| GPU | Single | 0.5 ms | 4,000× |
| GPU | Batch 10⁴ | <0.01 ms | >100,000× |

**Key insight**: "The most successful 'AI for Science' models are those that bake domain knowledge into the architecture or data preprocessing, rather than relying on the network to learn the laws of physics from scratch."

---

## Session: 2024-12-22 (Parallel Branch: ja-session-1059)

### 29. Unified Sensitivity Plots

Refactored parameter sensitivity plots to show all representations in a single figure:
- 2×3 grid: Amplitude, Phase, Time Domain / Real, Imaginary, Envelope
- Both physical units (Hz, ms) and geometric units (Mf, t/M) versions
- Fiducial parameter values displayed in title

### 30. XAS vs XPHM Comparison

Generated parallel sensitivity plots for IMRPhenomXAS to compare against XPHM:
- XAS: Aligned-spin, dominant (2,2) mode only
- XPHM: Higher modes + precession
- Same fiducial parameters for direct comparison
- Output: `figures/sensitivity/` (XPHM) and `figures/sensitivity_xas/` (XAS)

**Key observations:**
- Higher modes visible as "wiggles" in amplitude at high frequencies
- Most pronounced for unequal mass ratios and edge-on inclinations

### 31. Mode Selection Implementation

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

### 32. Mode Additivity Test

**Question:** Is h(f) = Σ_{lm} h_lm(f) for XPHM?

**Test:** Generated waveforms with individual mode groups and compared sum to full waveform.

**Results:**
```
Max |h_full - sum(h_lm)|:    1.4e-31 (machine precision)
Match(h_full, sum(h_lm)):    1.000000000000000
```

**Conclusion: ✓ Modes ARE additive** (to machine precision)

### 33. XPHM(2,2) vs XAS Comparison

**Question:** Does XPHM with only (2,±2) modes equal XAS?

**Test:** Generated both and computed match.

**Results:**
| Comparison | Match | Mismatch |
|------------|-------|----------|
| XAS vs XPHM (all modes) | 0.9966 | 3.4×10⁻³ |
| XAS vs XPHM (2,±2 only) | 0.9983 | 1.7×10⁻³ |

**Conclusion:** XPHM(2,2) is closer to XAS than XPHM(all), but **not identical** (mismatch ~1.7×10⁻³). This is because XAS and XPHM are separately calibrated models, not the same code with mode selection.

### 34. Important LAL Convention Discovery

**Finding:** When requesting individual modes in LAL:
- Positive m modes (2,2), (3,3), etc. include BOTH +m and -m contributions
- Negative m modes (2,-2), (3,-3), etc. return ZERO

This means for mode selection, use only positive m values:
```python
mode_array = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]  # Correct
# NOT: [(2, 2), (2, -2), (2, 1), (2, -1), ...]  # Redundant
```

### 35. Mode-by-Mode Sensitivity Plots

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

### 36. LAL Default Modes and Numerical Settings

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

### 37. Multibanding Test and Precision Mode Generation

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

## Session: 2024-12-22 (Lunch - Dulcedo, Eddington)

*Walked to Dulcedo café in the Eddington development for lunch (most of the department closed for the Christmas break on 22nd December). Consolidated findings and discussed strategy over food.*

### 38. Strategy Consolidation

Reviewed the day's progress and crystallized the implementation strategy:

**Core findings confirmed:**
- Modes decompose additively (verified to machine precision)
- Per-mode amplitude/phase are dramatically smoother than full waveform
- XPHM with only (2,2) mode gives XHM-like behavior for free

**Implementation strategy:**
- Start with (2,2) mode, aligned spins only (3D parameter space: η, χ₁z, χ₂z)
- Data generation is cheap (~tens of minutes for large sample)
- Build up mode by mode once (2,2) works
- η is the natural mass parameterization (not m₁, m₂) since we scale by total mass M

### 39. Literature Gap Analysis

Second deep research (seeded with mlgw_NN paper 2402.06587) revealed the actual PCA+NN lineage. Key observation: no good JAX emulator for XPHM currently exists in the literature.

**Additional papers found:**
- `2205.14066`: Useful decomposition of precessing modes, writes out spherical harmonics explicitly
- `2411.14893` (SEOBNRE-AI): Eccentric binaries, but mismatches ~10⁻² (poor)

**Key insight from mlgw_NN:** Inclination can be factored out by modeling spherical harmonics directly, reducing the parameter space further.

**Observation on existing work:** Several papers achieve only ~10⁻¹ to 10⁻³ mismatch. The consensus was that the core ideas are sound but implementations may be suboptimal - "they might just suck at machine learning."

---

## Session: 2024-12-22 (Afternoon - Theory Development)

### 40. General Waveform Decomposition Framework

Developed a comprehensive theoretical document addressing how to build a fully general neural network emulator that can handle arbitrary waveform complexity (precession, higher modes, eccentricity).

**Key Questions Addressed:**

1. **Where does inclination fit?** Inclination (ι) is an extrinsic parameter - enters only through spin-weighted spherical harmonics Y_{ℓm}(ι,φ₀). Neural network learns intrinsic modes; inclination handled analytically at inference.

2. **Is the coprecessing frame a restrictive assumption?** No - it's a coordinate transformation, not a physical assumption. Can be applied to any waveform including eccentric ones. What IS restrictive is the conjugate symmetry assumption made within that frame.

3. **Can we get eccentricity from XHM/XPHM?** No. Eccentricity affects intrinsic mode evolution. The coprecessing frame transformation handles precession, not eccentricity. Must start from an eccentric model (SEOBNRE, TEOBResumS-GIOTTO).

4. **How to handle Euler angles?** Proposed three architectural options:
   - Option 1: Extract then learn (classical → NN)
   - Option 2: Learn inertial modes directly
   - Option 3: Build Wigner rotation into network, learn decomposition end-to-end (recommended)

**Document Created:**
- `theory/general-waveform-decomposition.tex` (8 pages)
- `theory/general-waveform-decomposition.pdf`

**Key Equations:**
```
h(t; λ; ι, φ₀) = Σ_{ℓm} h_{ℓm}(t; λ) ⁻²Y_{ℓm}(ι, φ₀)           [spherical harmonic]
h_{ℓm}(t; λ) = Σ_{m'} D^ℓ_{mm'}(α,β,γ; t,λ) h^{cp}_{ℓm'}(t; λ)  [coprecessing rotation]
h^{cp}_{ℓm}(t; λ) = A_{ℓm}(t; λ) e^{i φ_{ℓm}(t; λ)}            [amplitude/phase]
```

### 13. External LLM Review

Submitted the theory document to OpenAI (GPT-5.2) and Gemini for critical review.

**Reviews stored:** `theory/reviews/openai-review.md`, `theory/reviews/gemini-review.md`

**Common Critical Feedback:**

1. **Gauge/Identifiability Problem in Option 3**: Both reviewers identified that the coprecessing decomposition is non-unique. Without regularization, network might set α=β=γ=0 and shove all oscillations into coprecessing modes, defeating the purpose.

2. **Angle Periodicity**: Should output sin(α), cos(α) rather than raw angles to avoid discontinuities at 2π wrapping.

3. **Missing Conventions**: Need to specify h = h₊ - ih× vs h₊ + ih×, Euler angle convention (ZYZ vs ZXZ), active vs passive rotation.

4. **Missing Citations**: Both noted the document should cite foundational NR surrogate work (Blackman et al. 2015/2017, Varma et al. 2019 NRSur7dq4, Boyle et al. 2011).

5. **Eccentricity Clarification**: "Eccentricity affects Euler angles" is only true when precession is also present.

**Suggested Improvements:**
- Add regularization term penalizing temporal variance in coprecessing modes
- Use sin/cos or quaternion representation for angles
- Specify time-series parameterization (basis coefficients, not raw functions)
- Add "Conventions" subsection

### Branch Created

`theory-decomposition` branch created from master for this work.

### Files Added

```
theory/
├── general-waveform-decomposition.tex
├── general-waveform-decomposition.pdf
├── .gitignore                          # LaTeX ephemera
└── reviews/
    ├── openai-review.md
    └── gemini-review.md
```

### 14. Iterative Review and Document Refinement

The theory document underwent iterative review using external LLMs (OpenAI GPT-5.2 and Gemini-3-pro) following the `iterative-review` skill workflow.

**Iteration 1 → 2 Changes (addressing initial reviews):**
- Added comprehensive Conventions section (§2): complex strain definition, ZYZ Euler angles, active rotation convention, Wigner D-matrix definition, inertial frame specification
- Fixed Wigner-D notation with explicit indices throughout
- Added phase parameter degeneracy note (φ₀ and φ_c partially degenerate)
- Expanded Option 3 with gauge-fixing: regularization term with 4 components, frame anchoring, initialization strategy
- Added NR Surrogate citations (Blackman 2015/2017, Varma 2019, Boyle 2011, Schmidt 2011)
- Changed Option 3 "convention dependence" from "No" to "Implicit†"

**Iteration 2 → 3 Changes (final fixes):**
- Fixed L_angle: changed ‖γ̈‖² to ‖γ̇‖² for consistency (penalize velocities, not acceleration)
- Fixed phase regularization: changed from ‖φ̈‖² to ‖ω̈‖² where ω = φ̇ is instantaneous frequency (regularizing raw phase curvature would fight the physical chirp)

**Final Review Status:**
| Reviewer | Iteration 2 | Iteration 3 |
|----------|-------------|-------------|
| Gemini-3-pro | APPROVED | - |
| GPT-5.2 | 2 fixes needed | APPROVED |

**Key equations in final document:**
```
L_reg = λ₁ L_mode + λ₂ L_angle + λ₃ L_minrot + λ₄ L_anchor

L_mode = Σ_ℓm (‖Ä_ℓm‖² + ‖ω̈_ℓm‖²)     # smooth modes (frequency, not phase!)
L_angle = ‖α̇‖² + ‖β̇‖² + ‖γ̇‖²          # slow angle evolution
L_minrot = ‖γ̇ + α̇ cos β‖²              # minimal rotation frame
L_anchor = α(t_ref)² + γ(t_ref)²        # frame anchoring
```

**Document statistics:**
- Final: 12 pages, ~510 lines
- Added: +165 lines from initial version
- Reviews stored: `theory/reviews/openai-review.md`, `theory/reviews/gemini-review.md`

### 15. Frequency-Domain Theory Document

The time-domain framework (§12-14) addresses fully general waveforms with precession, but the project's immediate target is IMRPhenomXHM—a frequency-domain model for non-precessing (aligned-spin) waveforms with higher harmonics. A dedicated frequency-domain document was written to address this case.

**Key insight**: In frequency domain, the waveform depends on the dimensionless product Mf (mass × frequency), not M and f separately. This follows from GR's scale invariance—there is no intrinsic mass scale in vacuum GR.

**Document created:** `theory/frequency-domain-emulation.tex` (14 pages)

**Key content:**
1. **Mass scaling**: Total mass M is not an intrinsic parameter. The network learns H_ℓm(Mf; η, χ₁, χ₂), and M only enters when converting to physical frequency at inference.

2. **Extrinsic parameters enter analytically**:
   - Distance: h̃ ∝ 1/D_L
   - Coalescence time: h̃(f; t_c) = h̃(f; 0) × exp(-2πift_c)
   - Coalescence phase: h̃_ℓm(f; φ_c) = h̃_ℓm(f; 0) × exp(-imφ_c)
   - Inclination/azimuth: via spherical harmonics Y_ℓm(ι, φ₀)

3. **No coprecessing frame needed** for non-precessing systems—just smooth amplitude Ã_ℓm(Mf) and phase Φ_ℓm(Mf) functions.

4. **Conjugate symmetry** for aligned spins: h̃_{ℓ,-m}(f) = (-1)^ℓ h̃*_ℓm(f), halving the modes to emulate.

5. **Representation choices**:
   - Amplitude: log-amplitude (handles 5 orders of magnitude)
   - Phase: direct unwrapped or PN-residual
   - Frequency grid: log-spaced in Mf

6. **Architecture**: PCA + MLP (CosmoPower-style) recommended

7. **Precession**: Handled via "twisting-up" approximation (separate from aligned-spin emulator)

8. **Validation**: Mismatch metric, target M < 10⁻³

**Comparison with time-domain framework:**

| Aspect | Time Domain | Frequency Domain |
|--------|-------------|------------------|
| Natural variable | t/M | Mf |
| Coprecessing frame | Yes (precessing) | No (approximated) |
| Extrinsic params | Mixed in | Analytic factors |
| Best for | Precessing, eccentric | Aligned-spin, circular |

---

### Current State

**Two theory documents:**

1. `theory/general-waveform-decomposition.tex` (12 pages, approved)
   - Time-domain framework for fully general waveforms
   - Coprecessing frame decomposition + Euler angle dynamics
   - End-to-end learning with regularization

2. `theory/frequency-domain-emulation.tex` (14 pages)
   - Frequency-domain framework for aligned-spin waveforms
   - Mass scaling (Mf as domain variable)
   - PCA + MLP architecture

**Files:**
```
theory/
├── general-waveform-decomposition.tex  # Time-domain (approved)
├── general-waveform-decomposition.pdf  # 12 pages
├── frequency-domain-emulation.tex      # Frequency-domain
├── frequency-domain-emulation.pdf      # 14 pages
├── .gitignore                          # LaTeX ephemera
└── reviews/
    ├── openai-review.md                # GPT-5.2 review
    └── gemini-review.md                # Gemini review
```

---

### Next Steps

1. ☑ Add multibanding control to wrapper (done)
2. ☑ Create data generation script (LAL XHM → HDF5 training data)
3. ☑ Implement PCA compression for amplitude/phase (per-mode)
4. ☑ Build neural network with Speculator activation (Flax)
5. ☑ Training pipeline with optax
6. ☐ Validation: mismatch < 10⁻³ target
7. ☐ Integration with ripple interface
8. ☑ XAS comparison plots (verify XPHM aligned-spin ≈ XAS)
9. ☒ Literature review: existing GW emulation approaches
10. ☒ General waveform decomposition theory document
11. ☒ Frequency-domain emulation theory document

**Theory:** Complete. Two complementary documents cover frequency-domain (immediate target) and time-domain (future precessing/eccentric extension).

---

## Session: 2024-12-22 (Evening - Mode Extraction Implementation)

### 16. Individual Mode Extraction

Implemented individual spherical harmonic mode extraction to align with the theory document's framework. Previously, the data generation script stored combined polarizations (h_+ and h_×), but the theory specifies learning individual modes h_ℓm(Mf).

**Key changes:**

1. **LAL Mode Extraction Functions** (`src/jim_emulators/waveforms/lal_waveforms.py`):
   - `generate_fd_mode(params, ell, emm)`: Extracts a single mode using `lalsim.SimIMRPhenomXHMGenerateFDOneMode`
   - `generate_fd_modes(params, modes)`: Helper for multiple modes
   - Returns raw complex h_ℓm(f) for amplitude/phase extraction

2. **Revised Data Generation** (`scripts/generate_data.py`):
   - Uses `generate_fd_mode()` instead of combined polarizations
   - Stores (2,2) mode by default, generalizable via `--mode` flag
   - Log-spaced Mf grid: [0.0005, 0.3] with 2000 points (extended from previous)
   - Phase aligned at reference frequency Mf_ref = 0.003 (not at peak amplitude)
   - Supports any mode: `--mode 3 3` for (3,3), etc.

3. **Data Validation Script** (`scripts/validate_data.py`):
   - Comprehensive checks: NaN analysis, amplitude/phase statistics, parameter coverage
   - Verifies stored data matches fresh LAL generation exactly
   - Generates validation plots

**Generated Dataset:**
```
data/waveforms_22mode.h5
├── frequency_grid          # 2000 log-spaced Mf points [0.0005, 0.3]
├── train/
│   ├── parameters          # [10000, 3] - (η, χ₁z, χ₂z)
│   ├── log_amplitude       # [10000, 2000] - log₁₀|h₂₂|
│   └── phase               # [10000, 2000] - unwrapped, aligned at Mf_ref
└── validation/
    └── ... (1000 samples)
```

**Validation Results:**
- 10,000 training + 1,000 validation samples generated in ~10 seconds
- Phase perfectly aligned at Mf_ref (deviation = 0)
- Data matches LAL exactly on verification (3 samples tested)
- All 42 tests pass (5 new tests added for mode extraction)

### 17. Frequency Grid Discussion

James noted that jim's likelihood inference may require linear grids in physical frequency f, while the theory recommends log-spaced grids in Mf for training. Current approach:

- **Training data**: Log-spaced Mf grid (better captures inspiral dynamics)
- **Inference**: Emulator can be evaluated at arbitrary Mf values (converted from physical f given source mass M)

This design allows the PCA+MLP architecture to learn on an optimal grid while supporting evaluation on any grid required by downstream inference code.

### Files Changed

```
src/jim_emulators/waveforms/lal_waveforms.py  # +128 lines (mode extraction)
src/jim_emulators/waveforms/__init__.py       # Export new functions
scripts/generate_data.py                       # Rewritten for mode extraction
scripts/validate_data.py                       # New validation script
tests/test_lal_waveforms.py                    # +5 tests for mode extraction
```

### Branch

`feature/mode-extraction` - committed as `d48805c`

### 18. No-Interpolation Fix

**Critical improvement**: Eliminated interpolation from data generation.

**Problem identified**: The initial implementation generated LAL waveforms on a uniform frequency grid, then interpolated to the target log-spaced Mf grid. This interpolation introduces errors and defeats the purpose of direct evaluation.

**Solution**: Found LAL's `SimIMRPhenomXHMFrequencySequenceOneMode` function which evaluates the waveform model at arbitrary frequency points without requiring a uniform grid.

**New function added** (`src/jim_emulators/waveforms/lal_waveforms.py`):
```python
def generate_fd_mode_at_frequencies(
    frequencies: np.ndarray,
    mass_1: float, mass_2: float,
    chi1z: float, chi2z: float,
    ell: int, emm: int,
    luminosity_distance: float = 1.0,
    phase: float = 0.0,
    f_ref: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a single mode h_lm evaluated at specified frequencies.
    Uses LAL's frequency sequence function for exact evaluation
    at arbitrary frequency points (no interpolation).
    """
```

**Validation results** (after fix):
```
Sample 0: eta=0.1465, chi1z=0.6249, chi2z=0.1810
  Amplitude: max diff=0.00e+00, mean=0.00e+00
  Phase: max diff=0.00e+00 rad, mean=0.00e+00
  ✓ Perfect match (machine precision)!
```

Stored data now matches fresh LAL generation to machine precision (zero difference), confirming the data generation pipeline is exact.

**Tests added**: 4 new tests in `TestFrequencySequenceGeneration` class:
- `test_generate_at_arbitrary_frequencies`
- `test_frequency_sequence_vs_uniform_grid`
- `test_log_spaced_grid_no_interpolation`
- `test_higher_modes_at_frequencies`

All 23 tests pass.

### 19. Phase Aliasing Fix

**Problem identified**: The unwrapped phase plot showed spurious oscillations at low frequencies. Investigation revealed phase aliasing - at very low Mf, the phase evolves so rapidly that it changes by more than 2π between adjacent log-spaced grid points, causing `np.unwrap` to fail.

**Diagnosis**:
- At Mf = 0.0005 (our original MF_MIN), phase derivative is extremely large
- With 2000 log-spaced points, adjacent frequencies differ by ~0.3%
- Phase was changing by >π between adjacent points for Mf < 0.003
- This created 270+ unwrapping failures in the problematic region

**Solution**: Increased `MF_MIN` from 0.0005 to 0.004 and `MF_REF` from 0.003 to 0.006.

Physical frequency coverage:
| Total Mass | f_min (MF_MIN=0.004) | f_ref (MF_REF=0.006) |
|------------|---------------------|---------------------|
| 50 M☉ | 16 Hz | 24 Hz |
| 20 M☉ | 40 Hz | 61 Hz |
| 10 M☉ | 81 Hz | 122 Hz |

This still covers the LIGO/Virgo sensitive band (>10-20 Hz) for typical BBH sources.

**Validation after fix**:
- Max phase jump between adjacent bins: 2.47 rad (< π)
- Phase correctly aligned at reference (deviation = 0)
- All samples pass verification at machine precision

### 20. External Review (OpenAI GPT-5.2)

Consulted OpenAI to validate the phase aliasing fix. Key feedback:

**Confirmation**: The fix is correct. Raising `MF_MIN` until the maximum adjacent phase jump is safely below π is a valid and commonly-used approach, provided the emulator doesn't need to represent that region.

**Mass range implications**:
| Total Mass | MF_MIN=0.004 corresponds to | Verdict |
|------------|----------------------------|---------|
| 50 M☉ | ~16 Hz | Fine (analyses often start at 15-20 Hz) |
| 20 M☉ | ~40 Hz | Lose 20-40 Hz band |
| 10 M☉ | ~81 Hz | Lose most of inspiral below 80 Hz |

For BBH with M ≳ 30-40 M☉ and typical f_min ~ 15-20 Hz: **our fix is appropriate**.
For lower-mass BBH (10-20 M☉) with f_min ~ 20 Hz: would need a different approach.

**Better approaches for future** (if we need low-Mf coverage):
1. **Grid in PN variable**: Sample uniformly in v = (πMf)^(1/3) instead of log(Mf) - phase is more polynomial in v
2. **Piecewise grid**: Dense at low Mf, log-spaced at higher Mf
3. **Residual phase**: Subtract a PN baseline phase, train on slowly-varying residual, add back at inference
4. **Train complex directly**: Use (Re(h), Im(h)) to bypass unwrapping entirely

**Recommendation for NN training**: Training on **residual phase** (subtract PN baseline) is usually the best balance for waveform emulators - reduces sampling density requirements and improves learning.

**Current status**: Our MF_MIN=0.004 fix is appropriate for the initial (2,2) mode emulator targeting typical BBH sources. For future extension to lower masses, consider residual phase or grid redesign.

### 21. Grid Comparison Study

Investigated whether using the PN variable v = (πMf)^(1/3) instead of log(Mf) for the frequency grid would help with phase aliasing.

**Finding 1: v-uniform grid doesn't help**

At Mf_min=0.004, comparing grids:
- Log(Mf) grid: max phase jump = 0.88 rad
- v-uniform grid: max phase jump = 1.95 rad

The log(Mf) grid is actually *better* because it concentrates points at low Mf where phase evolves fastest. The v-uniform grid spreads points too evenly.

**Finding 2: More points can extend to lower Mf_min**

Testing log(Mf) grid with varying N_FREQ and Mf_min:

| Mf_min | 1000 pts | 2000 pts | 4000 pts | 8000 pts | 16000 pts |
|--------|----------|----------|----------|----------|-----------|
| 0.0005 | 3.1 | 3.1 | 3.1 | 3.1 | 3.1 |
| 0.0010 | 3.1 | 3.1 | 3.1 | 2.9 | 1.5 |
| 0.0020 | 3.1 | 3.1 | 1.6 | 0.8 | 0.4 |
| 0.0030 | 3.0 | 1.5 | 0.8 | 0.4 | 0.2 |
| 0.0040 | 1.8 | 0.9 | 0.4 | 0.2 | 0.1 |

(Values are max phase jump in radians; need < π ≈ 3.14 for safe unwrapping)

**Practical recommendations:**

| Use Case | Mf_min | N_FREQ | f_min @ 50 M☉ | f_min @ 20 M☉ |
|----------|--------|--------|---------------|---------------|
| Current (M ≳ 40 M☉) | 0.004 | 2000 | 16 Hz | 41 Hz |
| Extended (M ≳ 20 M☉) | 0.002 | 4000 | 8 Hz | 20 Hz |
| Low-mass (M ≳ 10 M☉) | 0.001 | 8000 | 4 Hz | 10 Hz |

**Conclusion**: The log(Mf) grid is optimal. To support lower masses, increase N_FREQ rather than changing grid type. Our current setup (Mf_min=0.004, N_FREQ=2000) is appropriate for M ≳ 30-40 M☉ with f_min ~ 15-20 Hz.

### Files Changed

```
src/jim_emulators/waveforms/lal_waveforms.py  # +generate_fd_mode_at_frequencies()
src/jim_emulators/waveforms/__init__.py       # Export new function
scripts/generate_data.py                       # Use direct evaluation, fix MF_MIN/MF_REF
scripts/validate_data.py                       # Use direct evaluation for verification
tests/test_lal_waveforms.py                    # +4 tests for frequency sequence
```

---

### 22. Training Pipeline Documentation

Created comprehensive LaTeX documentation for the neural network training pipeline in `implementation/`.

**Document:** `implementation/training-pipeline.tex` (12 pages)

**Contents:**

1. **Data Characteristics** (§1) - Analysis of validation plots showing:
   - Log amplitude range: [-26, -20] (span of ~6)
   - Phase range: [-200, 0] radians (span of ~200)
   - Bimodal amplitude distribution (inspiral vs merger/ringdown)
   - Phase correctly aligned at Mf_ref = 0.006

2. **Normalization Strategy** (§2):
   - **Per-frequency standardization** for outputs: compute mean/std at each of the 2000 frequency bins, normalize independently
   - **Input normalization**: η → [-1,1] via linear map, χ → [-1,1] by dividing by 0.99
   - **Differentiability**: Normalization constants are precomputed and fixed, so gradients flow through trivially (just linear transformations)

3. **PCA Compression** (§3):
   - Separate PCA for amplitude and phase (different characteristics)
   - Target: 99.99% variance retention
   - Expected: 30-80 components per quantity

4. **Network Architecture** (§4):
   - 4×512 MLP with Speculator activation
   - Input: 3 normalized parameters (η̂, χ̂₁z, χ̂₂z)
   - Output: K_A + K_Φ PCA coefficients (~128 total)
   - ~850,000 parameters

5. **Training Configuration** (§5):
   - Loss: MSE on PCA coefficients
   - Optimizer: Adam with gradient clipping (global norm ≤ 1.0)
   - Learning rate: 10⁻³ with cosine decay to 10⁻⁵
   - Batch size: 256, epochs: 1000 with early stopping

6. **Inference Pipeline** (§6):
   - Full algorithm: params → normalize → MLP → PCA reconstruct → denormalize
   - Computational cost: ~1.1M FLOPs/waveform
   - Expected speedup: ~1000× over LALSimulation

7. **Validation Metrics** (§7):
   - Target mismatch: < 10⁻³
   - Amplitude error: < 1% relative
   - Phase error: < 0.1 radians

**Files created:**
```
implementation/
├── training-pipeline.tex         # Source (25 KB)
├── training-pipeline.pdf         # Compiled (12 pages)
└── waveforms_22mode_validation.png  # Reference plot
```

---

### Next Steps

1. ☑ Generate full 10,000 sample training dataset
2. ☐ Implement PCA compression (sklearn fit, JAX transform)
3. ☐ Train emulator on (2,2) mode data
4. ☐ Validate mismatch < 10⁻³
5. ☐ Extend to higher modes (2,1), (3,3), etc.
6. ☐ Address linear frequency grid for jim integration if needed
