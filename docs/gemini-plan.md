This is a detailed plan of attack for building **Jim Emulators**, a JAX-based waveform emulation library. This plan synthesizes the architectural decisions from **CosmoPower/Speculator**, the physics handling from **ripple**, and the specific requirements of the **jim** ecosystem.

---

## 1. Architecture Analysis

We will adopt the **PCA-plus-NN** strategy used in CosmoPower and Speculator. This reduces the dimensionality of the output space (frequency bins) to a manageable number of latent coefficients.

### A. The Neural Network (Flax/JAX)
Based on `cosmopower_NN.py` and the *Speculator* paper (1911.11778), the network topology is a Multi-Layer Perceptron (MLP) with a specialized activation function.

*   **Framework**: JAX + Flax (Linen API).
*   **Input**: Normalized intrinsic parameters $\theta$ (dimension $D_{\text{in}}$).
*   **Hidden Layers**: 4 layers of 512 units (standard CosmoPower config).
*   **Activation Function**: The "Speculator" activation (Eq. 4 in 1911.11778). This is critical for smooth derivatives required for HMC.
    $$ \sigma(\mathbf{x}) = [\boldsymbol{\gamma} + (1 + e^{-\boldsymbol{\beta} \odot \mathbf{x}})^{-1} (1 - \boldsymbol{\gamma})] \odot \mathbf{x} $$
    *   $\boldsymbol{\beta}$ and $\boldsymbol{\gamma}$ are **trainable parameters** vectors of the same size as the layer width.
*   **Output**: PCA coefficients $\boldsymbol{\alpha}$ (dimension $N_{\text{PCA}}$).

### B. Preprocessing & Compression
We cannot emulate raw strain $h(f)$ directly due to phase wrapping and dynamic range. We will emulate **Amplitude** and **Phase** separately.

1.  **Decomposition**: $h(f) = A(f)e^{-i\Phi(f)}$.
2.  **Log-Amplitude**: $Y_A = \log_{10}(A(f))$.
3.  **Unwrapped Phase**: $Y_\Phi = \text{unwrap}(\Phi(f))$.
    *   *Note*: We must remove the linear trend ($t_c$) and constant offset ($\phi_c$) before training, or align waveforms such that peak is at $t=0$. Ripple does this alignment analytically; we must ensure training data is aligned similarly.
4.  **Standardization**: $Y' = (Y - \mu_Y) / \sigma_Y$.
5.  **PCA**: Perform SVD on the training batch of $Y'$ to obtain basis vectors $\mathbf{V}$. Keep enough components to capture $99.99\%$ variance (typically 20-60 components).

---

## 2. Data Generation Pipeline

We will generate training data using `lalsuite`, storing it in a format that allows the emulator to learn the **intrinsic** manifold.

### A. Grid Strategy
We emulate in the **geometric frequency domain** ($M \cdot f$) to make the waveform mass-independent (up to amplitude scaling).

*   **Waveform**: `IMRPhenomXAS` (Phase 1: Aligned Spin).
*   **Frequency Grid**: Log-spaced grid in dimensionless units $x = M f$.
    *   Range: $x \in [0.003, 0.25]$ (Covers early inspiral to ringdown).
    *   Points: $N_f = 1000$ (High resolution needed for accurate PCA).
*   **Parameter Space (Intrinsic)**:
    *   $\eta \in [0.05, 0.25]$ (Symmetric mass ratio).
    *   $\chi_{1z}, \chi_{2z} \in [-0.99, 0.99]$ (Dimensionless spins).
*   **Sampling**: Latin Hypercube Sampling (LHS) via `scipy.stats.qmc`.
    *   Training Set: $10^5$ samples.
    *   Validation Set: $10^4$ samples.

### B. Storage Format
HDF5 is preferred for large datasets.

**File:** `data/training_waveforms.h5`
*   `/parameters`: Shape $(N, 3)$ [$\eta, \chi_{1z}, \chi_{2z}$]
*   `/amplitude`: Shape $(N, N_f)$ [$\log_{10}|h|$]
*   `/phase`: Shape $(N, N_f)$ [Unwrapped, peak-aligned phase]
*   `/frequency_grid`: Shape $(N_f,)$ [Geometric frequencies $Mf$]

**Script:** `scripts/generate_data.py`
```python
import lalsimulation as lalsim
# Logic: Loop over LHS params -> Gen LAL waveform -> Convert to Geometric Units
# -> Compute Amp/Phase -> Unwrap Phase -> Save
```

---

## 3. Emulator Implementation Plan

We will build a package named `jim_emulators`.

### Step 3.1: PCA Logic (`src/jim_emulators/components/pca.py`)
Implement a class `WaveformPCA` similar to `cosmopower_PCA.py` but utilizing JAX for the projection step (for differentiability later).
*   Methods: `fit(data)`, `transform(data)`, `inverse_transform(coeffs)`.
*   Must save `mean`, `std`, and `components_` (the V matrix).

### Step 3.2: The Neural Network (`src/jim_emulators/components/network.py`)
Implement the Speculator architecture using `flax.linen`.

```python
import flax.linen as nn

class SpeculatorActivation(nn.Module):
    def setup(self):
        # Learnable parameters per neuron
        self.beta = self.param('beta', nn.initializers.ones, (1,)) 
        self.gamma = self.param('gamma', nn.initializers.zeros, (1,))

    def __call__(self, x):
        # Implementation of Eq. 4 from Speculator paper
        sig = tf.sigmoid(self.beta * x) # JAX equiv
        return (self.gamma + (1 - self.gamma) * sig) * x

class EmulatorNet(nn.Module):
    n_layers: int
    n_units: int
    n_outputs: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(self.n_units)(x)
            x = SpeculatorActivation()(x)
        x = nn.Dense(self.n_outputs)(x)
        return x
```

### Step 3.3: The Interface (`src/jim_emulators/emulator.py`)
A wrapper class `GWEmu` that stitches preprocessing, NN, and PCA reconstruction.
*   `__init__`: Loads weights and PCA basis.
*   `predict(theta)`: 
    1.  Normalize `theta`.
    2.  NN Forward pass $\to$ `coeffs`.
    3.  PCA Inverse Transform $\to$ `standardized_y`.
    4.  Un-standardize $\to$ `log_amp` and `phase`.
    5.  Return `10**log_amp`, `phase`.

---

## 4. Training Pipeline

**Script:** `scripts/train.py`

1.  **Load Data**: Read HDF5.
2.  **PCA compression**: Fit PCA on amplitude and phase *separately*. Save basis matrices.
    *   Target: ~99.99% explained variance. Expect ~20-40 components for Amplitude, ~10-20 for Phase.
3.  **Prepare Tensors**: Create $(X, Y_{pca})$ pairs.
4.  **Training Loop**:
    *   **Loss**: MSE Loss on PCA coefficients.
        $$ L = \frac{1}{N} \sum || \mathbf{\alpha}_{pred} - \mathbf{\alpha}_{true} ||^2 $$
    *   **Optimizer**: `optax.adamw(learning_rate=1e-3)`.
    *   **Scheduler**: `optax.warmup_cosine_decay_schedule` (Warmup 10% epochs, decay to 1e-6).
    *   **Batch Size**: 256 or 512.
    *   **Epochs**: 200-500 with Early Stopping based on validation loss.

---

## 5. Validation & Benchmarking

**Script:** `scripts/validate.py`

We quantify accuracy using the **Match** (overlap) metric, utilizing `ripple`'s utility functions.

1.  **Test Set**: Use the 10k hold-out set.
2.  **Reconstruction**:
    *   Predict $A_{emu}, \Phi_{emu}$.
    *   Construct $h_{emu}(f) = A_{emu}(f) e^{-i\Phi_{emu}(f)}$.
3.  **Mismatch Calculation**:
    *   Use `ripple.get_match_arr`.
    *   Compute mismatch $\mathcal{M} = 1 - \langle h_{lal} | h_{emu} \rangle$.
4.  **Targets**:
    *   Median mismatch $< 10^{-4}$.
    *   Worst-case mismatch $< 10^{-2}$.
5.  **Plots**:
    *   Histogram of mismatches.
    *   Scatter plot: Mismatch vs Parameters (identify weak regions).
    *   Direct waveform comparison (Amplitude/Phase residuals) for worst offenders.

---

## 6. Integration into Jim/Ripple

This is the final step to make the emulator usable in inference.

**File:** `src/jim_emulators/interface.py`

We need a function signature compatible with `ripple` likelihoods.

```python
def gen_IMRPhenomXAS_emu(f, theta, f_ref):
    """
    f: Frequency grid (JAX array)
    theta: [Mc, eta, chi1, chi2, D_L, tc, phic, inclination]
    """
    # 1. Parameter Conversion
    # Convert Mc, eta -> M_total, eta
    # Extract spins
    
    # 2. Get Geometric Frequency Grid
    # The emulator was trained on a fixed Mf grid.
    # We need to predict on that grid, then interpolate to the requested `f`.
    
    # 3. Emulator Prediction (JIT-compiled)
    # amp_geo, phase_geo = emulator.predict(intrinsic_params)
    
    # 4. Interpolation
    # M_tot = ...
    # f_geo = f * M_tot * (G/c^3)
    # amp = jnp.interp(f_geo, emulator.training_grid, amp_geo)
    # phase = jnp.interp(f_geo, emulator.training_grid, phase_geo)
    
    # 5. Extrinsic Scaling
    # Scale Amplitude by M_tot^2 / D_L
    # Add time/phase shifts to Phase: phase += 2*pi*f*tc - phic
    
    # 6. Polarizations
    # h0 = amp * exp(-1j * phase)
    # hp = h0 * (1 + cos^2(iota))/2
    # hc = -1j * h0 * cos(iota)
    
    return hp, hc
```

### Action Items Checklist

1.  [ ] **Repo Init**: Setup structure, requirements (jax, flax, lalsuite, h5py).
2.  [ ] **Data Gen**: Write `generate_data.py` (LAL $\to$ H5).
3.  [ ] **Components**: Implement `WaveformPCA` and `EmulatorNet` (w/ Speculator activation).
4.  [ ] **Training**: Train Aligned-Spin (XAS) model on Amplitude and Phase.
5.  [ ] **Validation**: Run mismatch tests against LAL.
6.  [ ] **Integration**: Write the `gen_IMRPhenomXAS_emu` wrapper and test differentiability.