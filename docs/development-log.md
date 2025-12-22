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

---

### Next Steps

Per `gemini-plan.md` (updated with XPHM decision):
1. Repository structure setup (src/jim_emulators/)
2. Data generation script (LAL XPHM → HDF5)
3. PCA and network components (Flax/JAX)
4. Training pipeline
5. Validation against LAL
6. Integration with ripple interface
