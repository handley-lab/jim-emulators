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

---

## Session: 2025-12-22 (Theory Development)

### 12. General Waveform Decomposition Framework

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

Per `gemini-plan.md` (updated with XPHM decision):
1. Repository structure setup (src/jim_emulators/)
2. Data generation script (LAL XHM → HDF5) — frequency domain, aligned-spin
3. PCA and network components (Flax/JAX)
4. Training pipeline
5. Validation against LAL (mismatch metric)
6. Integration with ripple interface

**Theory:** Complete. Two complementary documents cover frequency-domain (immediate target) and time-domain (future precessing/eccentric extension).
