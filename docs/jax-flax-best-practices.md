# JAX/Flax Best Practices for Scientific Emulators

This document synthesizes code review feedback from GPT-5 and Gemini on our Flax implementation, focusing on best practices for gravitational wave waveform emulation.

> **Review date:** 2024-12-22
> **Code reviewed:** `snippets/flax_nn.py`, `scripts/train_toy_example.py`

---

## 1. SpeculatorActivation Implementation

### Current Implementation
```python
class SpeculatorActivation(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        alpha = self.param('alpha', nn.initializers.ones, (self.features,))
        gamma = self.param('gamma', nn.initializers.zeros, (self.features,))
        sigmoid_term = jax.nn.sigmoid(alpha * x)
        return (gamma + sigmoid_term * (1.0 - gamma)) * x
```

### Issues Identified

**1. Unconstrained `gamma` (OpenAI):**
- The Speculator paper requires γ ∈ [0,1] as a convex mixer between linear and gated behavior
- Our implementation allows γ to go negative or >1, changing qualitative behavior

**2. Pass `features` explicitly (Gemini):**
- Flax idiom prefers "lazy initialization" (shape inference from input)
- Reduces configuration boilerplate and mismatch errors

### Recommended Implementation

```python
class SpeculatorActivation(nn.Module):
    """
    Speculator activation with constrained parameters and shape inference.

    σ(x) = [γ + (1-γ)·sigmoid(α·x)]·x

    Where γ ∈ (0,1) is enforced via sigmoid on learnable logits.
    """
    @nn.compact
    def __call__(self, x):
        features = x.shape[-1]  # Lazy shape inference

        # α controls sigmoid steepness (unconstrained, or use softplus for positive)
        alpha = self.param('alpha', nn.initializers.ones, (features,))

        # γ stored as logits, mapped through sigmoid to (0,1)
        # Initialize to -2.0 (mostly gated) - see empirical note below
        gamma_logits = self.param('gamma_logits',
                                   nn.initializers.constant(-2.0), (features,))
        gamma = jax.nn.sigmoid(gamma_logits)

        sigmoid_term = jax.nn.sigmoid(alpha * x)
        return (gamma + (1.0 - gamma) * sigmoid_term) * x
```

**Initialization Note (Empirical):** We tested different gamma_logits initializations:
- **-2.0 (mostly gated):** sigmoid(-2) ≈ 0.12 → 1330x improvement, final loss 0.000349
- **+2.0 (mostly linear):** sigmoid(+2) ≈ 0.88 → 176x improvement, final loss 0.003791

The mostly gated initialization significantly outperforms the linear start on our toy task, contrary to some theoretical expectations. The gated start allows the network to learn complex nonlinear mappings more effectively.

**Optional:** Constrain α to be positive via `softplus(alpha_raw) + eps` if "steepness" semantics matter.

---

## 2. Normalization Handling

### Issue (Gemini)
Storing normalization statistics as Python class attributes:
- Won't be saved with model checkpoints
- Can cause JIT recompilation issues with large arrays

### Best Practice: Use Flax Variables

```python
class WaveformEmulator(nn.Module):
    n_hidden: int = 4
    n_units: int = 512
    n_outputs: int = 50

    @nn.compact
    def __call__(self, x):
        # Store normalization as non-trainable variables (survives checkpointing)
        in_mean = self.variable('constants', 'in_mean',
                                lambda: jnp.zeros(x.shape[-1]))
        in_std = self.variable('constants', 'in_std',
                               lambda: jnp.ones(x.shape[-1]))
        out_mean = self.variable('constants', 'out_mean',
                                 lambda: jnp.zeros(self.n_outputs))
        out_std = self.variable('constants', 'out_std',
                                lambda: jnp.ones(self.n_outputs))

        # Apply input normalization
        x = (x - in_mean.value) / in_std.value

        # MLP backbone
        for i in range(self.n_hidden):
            x = nn.Dense(self.n_units)(x)
            x = SpeculatorActivation()(x)  # No features arg needed

        x = nn.Dense(self.n_outputs)(x)

        # Apply output denormalization
        return x * out_std.value + out_mean.value
```

---

## 3. Training Loop Performance

### Issues Identified

**1. Python batch loop (OpenAI):**
- Current: Python loop calls jitted `train_step` per batch
- Dispatch overhead, especially on GPU/TPU
- Better: JIT the whole epoch with `lax.scan`

**2. Data dropping (Gemini):**
- `n_batches = n_samples // batch_size` discards remainder
- Fix: Round up and handle partial last batch

**3. Host sync overhead (OpenAI):**
- `float(loss)` forces synchronization each epoch
- Log less frequently or batch the conversion

### Recommended: `lax.scan` Epoch

```python
def make_train_epoch(batch_size):
    @jax.jit
    def train_epoch(state, x, y, perm):
        x, y = x[perm], y[perm]
        n_batches = x.shape[0] // batch_size

        # Reshape into batches
        x = x[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
        y = y[:n_batches * batch_size].reshape(n_batches, batch_size, -1)

        def step_fn(state, batch):
            bx, by = batch
            state, loss = train_step(state, bx, by)
            return state, loss

        state, losses = jax.lax.scan(step_fn, state, (x, y))
        return state, losses.mean()

    return train_epoch
```

### Buffer Donation (OpenAI)

Reduce memory and improve speed:
```python
@partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch_x, batch_y):
    ...
```

---

## 4. Optimizer Configuration

### Recommended: Gradient Clipping + Schedule

```python
def create_optimizer(learning_rate, warmup_steps, total_steps):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=1e-6
    )

    return optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
        optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    )
```

---

## 5. PCA Implementation

### Recommendation (Both)

- **Fitting:** Use `sklearn.decomposition.PCA` (numerically stable SVD)
- **Transform/Inverse:** Implement in JAX for differentiability

```python
# Fit with sklearn (offline)
from sklearn.decomposition import PCA
pca = PCA(n_components=50).fit(training_data)

# Store basis for JAX
pca_components = jnp.array(pca.components_)  # (n_components, n_features)
pca_mean = jnp.array(pca.mean_)

# JAX transform (differentiable)
def pca_transform(x, components, mean):
    return (x - mean) @ components.T

def pca_inverse_transform(coeffs, components, mean):
    return coeffs @ components + mean
```

### End-to-End Differentiable Emulator

Inject PCA basis as constant for full gradient flow (needed for HMC):

```python
class EndToEndEmulator(nn.Module):
    pca_basis: jnp.ndarray  # Shape (n_components, n_freq), frozen

    @nn.compact
    def __call__(self, params):
        # NN predicts PCA coefficients
        coeffs = EmulatorMLP()(params)

        # Reconstruct waveform (differentiable)
        waveform = coeffs @ self.pca_basis
        return waveform
```

---

## 6. Architecture Suggestions

### Consider LayerNorm (OpenAI)
For stability with scientific data having varying scales:
```python
x = nn.Dense(self.n_units)(x)
x = nn.LayerNorm()(x)  # Add normalization
x = SpeculatorActivation()(x)
```

### Separate Heads for Amplitude/Phase (OpenAI)
Amplitude and phase have different scales/statistics. Consider:
- Two output heads with separate normalization
- Or two separate networks

---

## 7. Precision Considerations

### Current: Global float64
```python
jax.config.update("jax_enable_x64", True)
```

This is necessary for GW phase accuracy and HMC gradients.

### Performance Optimization (OpenAI)
For faster training:
- Train in float32 (fast GPU)
- Evaluate/HMC in float64 (accurate)

```python
# Training
params_f32 = jax.tree_map(lambda x: x.astype(jnp.float32), params)

# Inference/HMC
params_f64 = jax.tree_map(lambda x: x.astype(jnp.float64), params)
```

---

## 8. Checkpointing

### Use Orbax (Modern JAX Standard)

```python
import orbax.checkpoint as ocp

checkpointer = ocp.PyTreeCheckpointer()

# Save
checkpointer.save('/path/to/checkpoint', state)

# Restore
state = checkpointer.restore('/path/to/checkpoint', item=state)
```

---

## 9. Alternative Frameworks

### Equinox (OpenAI mention)
For scientific emulators with lots of custom differentiable physics, Equinox can be more ergonomic (pytrees + easier custom modules). Flax is fine for our use case with TrainState patterns.

---

## Summary: Priority Fixes

| Priority | Issue | Fix |
|----------|-------|-----|
| **High** | Unconstrained γ | Use sigmoid on logits |
| **High** | Normalization not checkpointed | Use Flax variables |
| **Medium** | Python batch loop | Use `lax.scan` |
| **Medium** | No gradient clipping | Add `optax.clip_by_global_norm` |
| **Medium** | No checkpointing | Add Orbax |
| **Low** | Explicit features arg | Use shape inference |
| **Low** | Float64 training speed | Consider mixed precision |

---

## References

- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)
- [Orbax Checkpointing](https://orbax.readthedocs.io/)
- Speculator paper: arXiv:1911.11778
- CosmoPower paper: arXiv:2106.03846
