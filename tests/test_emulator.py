"""
Tests for the waveform emulator components.

Run with: pytest tests/test_emulator.py -v
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import tempfile

jax.config.update("jax_enable_x64", True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jim_emulators.emulator import WaveformPCA, SpeculatorActivation, EmulatorMLP, GWEmulator
from jim_emulators.emulator.emulator import create_train_state, train_step, eval_loss


# =============================================================================
# PCA Tests
# =============================================================================

class TestWaveformPCA:
    """Tests for WaveformPCA class."""

    def test_fit_basic(self):
        """Test basic PCA fitting."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 50))

        pca = WaveformPCA(n_components=10)
        pca.fit(X)

        assert pca.n_components == 10
        assert pca.components_.shape == (10, 50)
        assert pca.mean_.shape == (50,)
        assert pca.std_.shape == (50,)

    def test_fit_auto_components(self):
        """Test automatic component selection by explained variance."""
        rng = np.random.default_rng(42)
        # Create data with clear low-rank structure
        U = rng.random((100, 5))
        V = rng.random((5, 50))
        X = U @ V + 0.01 * rng.random((100, 50))

        pca = WaveformPCA(explained_variance_target=0.99)
        pca.fit(X)

        # Should select ~5 components for this low-rank data
        assert pca.n_components <= 10
        assert np.sum(pca.explained_variance_ratio_) >= 0.99

    def test_transform_inverse_transform(self):
        """Test transform and inverse_transform are consistent."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 50))

        pca = WaveformPCA(n_components=40)  # Keep most variance
        pca.fit(X)

        coeffs = pca.transform(X)
        X_recon = pca.inverse_transform(coeffs)

        # With 40/50 components, reconstruction should be very good
        mse = np.mean((X - X_recon) ** 2)
        assert mse < 0.01

    def test_fit_transform(self):
        """Test fit_transform equals fit then transform."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 50))

        pca1 = WaveformPCA(n_components=10)
        coeffs1 = pca1.fit_transform(X)

        pca2 = WaveformPCA(n_components=10)
        pca2.fit(X)
        coeffs2 = pca2.transform(X)

        np.testing.assert_allclose(coeffs1, coeffs2)

    def test_reconstruction_error(self):
        """Test reconstruction_error calculation."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 50))

        pca = WaveformPCA(n_components=10)
        pca.fit(X)

        mse, per_sample_mse = pca.reconstruction_error(X)

        assert mse > 0
        assert per_sample_mse.shape == (100,)
        np.testing.assert_almost_equal(mse, np.mean(per_sample_mse))

    def test_jax_compatibility(self):
        """Test JAX-compatible methods work with JIT."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 50))

        pca = WaveformPCA(n_components=10)
        pca.fit(X)

        # Convert to JAX arrays
        X_jax = jnp.array(X[:10])

        # Test transform_jax
        coeffs_jax = pca.transform_jax(X_jax)
        coeffs_np = pca.transform(X[:10])
        np.testing.assert_allclose(np.array(coeffs_jax), coeffs_np, rtol=1e-5)

        # Test inverse_transform_jax
        X_recon_jax = pca.inverse_transform_jax(coeffs_jax)
        X_recon_np = pca.inverse_transform(coeffs_np)
        np.testing.assert_allclose(np.array(X_recon_jax), X_recon_np, rtol=1e-5)

    def test_serialization(self):
        """Test to_dict and from_dict."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 50))

        pca = WaveformPCA(n_components=10)
        pca.fit(X)

        # Serialize and deserialize
        d = pca.to_dict()
        pca2 = WaveformPCA.from_dict(d)

        # Check equality
        assert pca.n_components == pca2.n_components
        np.testing.assert_allclose(pca.components_, pca2.components_)
        np.testing.assert_allclose(pca.mean_, pca2.mean_)
        np.testing.assert_allclose(pca.std_, pca2.std_)


# =============================================================================
# Network Tests
# =============================================================================

class TestSpeculatorActivation:
    """Tests for SpeculatorActivation."""

    def test_output_shape(self):
        """Test output has correct shape."""
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((10, 64))

        act = SpeculatorActivation()
        params = act.init(rng, x)
        y = act.apply(params, x)

        assert y.shape == x.shape

    def test_smooth_gradients(self):
        """Test gradients are smooth (critical for HMC)."""
        rng = jax.random.PRNGKey(0)

        act = SpeculatorActivation()
        x = jnp.linspace(-3, 3, 100).reshape(1, -1)
        params = act.init(rng, x)

        # Compute gradients
        def forward(x):
            return act.apply(params, x).sum()

        grad_fn = jax.grad(forward)
        grads = grad_fn(x)

        # Gradients should exist and be finite
        assert jnp.all(jnp.isfinite(grads))

        # Check gradient is smooth (no sudden jumps)
        grad_diff = jnp.diff(grads.flatten())
        assert jnp.max(jnp.abs(grad_diff)) < 1.0  # No sharp transitions

    def test_learnable_parameters(self):
        """Test that alpha and gamma_logits are learnable."""
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((10, 64))

        act = SpeculatorActivation()
        params = act.init(rng, x)

        # Check parameters exist
        assert 'alpha' in params['params']
        assert 'gamma_logits' in params['params']
        assert params['params']['alpha'].shape == (64,)
        assert params['params']['gamma_logits'].shape == (64,)


class TestEmulatorMLP:
    """Tests for EmulatorMLP."""

    def test_output_shape(self):
        """Test output has correct shape."""
        rng = jax.random.PRNGKey(0)
        model = EmulatorMLP(n_hidden=2, n_units=64, n_outputs=30)

        x = jnp.ones((10, 3))
        params = model.init(rng, x)
        y = model.apply(params, x)

        assert y.shape == (10, 30)

    def test_parameter_count(self):
        """Test reasonable parameter count."""
        rng = jax.random.PRNGKey(0)
        model = EmulatorMLP(n_hidden=4, n_units=512, n_outputs=50)

        x = jnp.ones((1, 3))
        params = model.init(rng, x)

        n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

        # Rough estimate: 4 layers of 512, input 3, output 50
        # Each dense: weights + bias + activation params
        assert n_params > 500000  # Should have many parameters
        assert n_params < 2000000  # But not too many

    def test_differentiable(self):
        """Test network is differentiable."""
        rng = jax.random.PRNGKey(0)
        model = EmulatorMLP(n_hidden=2, n_units=64, n_outputs=30)

        x = jnp.ones((1, 3))
        params = model.init(rng, x)

        def forward(x):
            return model.apply(params, x).sum()

        grad_fn = jax.grad(forward)
        grads = grad_fn(x)

        assert grads.shape == x.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_jit_compatible(self):
        """Test network works with JIT compilation."""
        rng = jax.random.PRNGKey(0)
        model = EmulatorMLP(n_hidden=2, n_units=64, n_outputs=30)

        x = jnp.ones((10, 3))
        params = model.init(rng, x)

        @jax.jit
        def forward(x):
            return model.apply(params, x)

        y = forward(x)
        assert y.shape == (10, 30)


# =============================================================================
# Training Tests
# =============================================================================

class TestTraining:
    """Tests for training utilities."""

    def test_create_train_state(self):
        """Test training state creation."""
        rng = jax.random.PRNGKey(0)
        model = EmulatorMLP(n_hidden=2, n_units=64, n_outputs=30)

        state = create_train_state(rng, model, learning_rate=1e-3, input_dim=3)

        assert state.params is not None
        assert state.tx is not None

    def test_train_step(self):
        """Test single training step."""
        rng = jax.random.PRNGKey(0)
        model = EmulatorMLP(n_hidden=2, n_units=64, n_outputs=30)

        state = create_train_state(rng, model, learning_rate=1e-3, input_dim=3)

        # Dummy data
        batch_x = jnp.ones((32, 3))
        batch_y = jnp.zeros((32, 30))

        state_new, loss = train_step(state, batch_x, batch_y)

        assert loss > 0
        assert state_new.step == state.step + 1

    def test_loss_decreases(self):
        """Test that loss decreases during training."""
        rng = jax.random.PRNGKey(0)
        model = EmulatorMLP(n_hidden=2, n_units=64, n_outputs=30)

        state = create_train_state(rng, model, learning_rate=1e-2, input_dim=3)

        # Simple target function
        batch_x = jax.random.uniform(rng, (100, 3))
        batch_y = jnp.sin(batch_x[:, 0:1]) * jnp.ones((100, 30))

        initial_loss = float(eval_loss(state, batch_x, batch_y))

        # Train for a few steps
        for i in range(50):
            rng, step_rng = jax.random.split(rng)
            state, _ = train_step(state, batch_x, batch_y)

        final_loss = float(eval_loss(state, batch_x, batch_y))

        assert final_loss < initial_loss


# =============================================================================
# Emulator Integration Tests
# =============================================================================

class TestGWEmulator:
    """Tests for complete GWEmulator."""

    @pytest.fixture
    def mock_emulator(self):
        """Create a mock emulator for testing."""
        rng = jax.random.PRNGKey(0)

        # Create simple PCA
        pca_amp = WaveformPCA(n_components=10)
        pca_phase = WaveformPCA(n_components=10)

        # Fit with random data
        rng_np = np.random.default_rng(42)
        pca_amp.fit(rng_np.random((100, 50)))
        pca_phase.fit(rng_np.random((100, 50)))

        # Create network
        model = EmulatorMLP(n_hidden=2, n_units=64, n_outputs=20)
        params = model.init(rng, jnp.ones((1, 3)))

        # Create emulator
        return GWEmulator(
            network=model,
            network_params=params,
            pca_amplitude=pca_amp,
            pca_phase=pca_phase,
            frequency_grid=np.linspace(0.003, 0.25, 50),
            params_mean=np.array([0.15, 0.0, 0.0]),
            params_std=np.array([0.1, 0.5, 0.5]),
        )

    def test_predict_shape(self, mock_emulator):
        """Test predict returns correct shapes."""
        params = np.array([[0.15, 0.3, -0.2]])
        log_amp, phase = mock_emulator.predict(params)

        assert log_amp.shape == (1, 50)
        assert phase.shape == (1, 50)

    def test_predict_batch(self, mock_emulator):
        """Test predict works with batches."""
        params = np.random.rand(10, 3)
        params[:, 0] = params[:, 0] * 0.2 + 0.05  # eta in [0.05, 0.25]

        log_amp, phase = mock_emulator.predict(params)

        assert log_amp.shape == (10, 50)
        assert phase.shape == (10, 50)

    def test_predict_strain(self, mock_emulator):
        """Test predict_strain returns complex strain."""
        params = np.array([[0.15, 0.3, -0.2]])
        strain = mock_emulator.predict_strain(params)

        assert strain.shape == (1, 50)
        assert np.iscomplexobj(strain)

    def test_save_load(self, mock_emulator):
        """Test save and load round-trip."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            mock_emulator.save(path)
            loaded = GWEmulator.load(path)

            # Test predictions match
            params = np.array([[0.15, 0.3, -0.2]])
            log_amp1, phase1 = mock_emulator.predict(params)
            log_amp2, phase2 = loaded.predict(params)

            np.testing.assert_allclose(log_amp1, log_amp2)
            np.testing.assert_allclose(phase1, phase2)
        finally:
            Path(path).unlink()

    def test_jit_compilation(self, mock_emulator):
        """Test that prediction is JIT-compiled."""
        params = np.array([[0.15, 0.3, -0.2]])

        # First call triggers compilation
        _ = mock_emulator.predict(params)

        # Second call should be fast (JIT cached)
        import time
        start = time.time()
        for _ in range(100):
            _ = mock_emulator.predict(params)
        elapsed = time.time() - start

        # 100 predictions should be very fast (< 1 second)
        assert elapsed < 1.0


# =============================================================================
# Gradient Tests (Critical for HMC)
# =============================================================================

class TestGradients:
    """Tests for gradient computation (critical for HMC inference)."""

    def test_emulator_gradients(self):
        """Test gradients through full emulator pipeline."""
        rng = jax.random.PRNGKey(0)

        # Simple setup
        pca_amp = WaveformPCA(n_components=5)
        pca_phase = WaveformPCA(n_components=5)

        rng_np = np.random.default_rng(42)
        pca_amp.fit(rng_np.random((50, 20)))
        pca_phase.fit(rng_np.random((50, 20)))

        model = EmulatorMLP(n_hidden=2, n_units=32, n_outputs=10)
        params = model.init(rng, jnp.ones((1, 3)))

        # Define gradient computation
        def compute_strain_norm(input_params, network_params):
            input_norm = (input_params - jnp.array([0.15, 0.0, 0.0])) / jnp.array([0.1, 0.5, 0.5])
            pca_coeffs = model.apply(network_params, input_norm)

            amp_coeffs = pca_coeffs[:, :5]
            phase_coeffs = pca_coeffs[:, 5:]

            log_amp = pca_amp.inverse_transform_jax(amp_coeffs)
            phase = pca_phase.inverse_transform_jax(phase_coeffs)

            amp = 10.0 ** log_amp
            strain = amp * jnp.exp(-1j * phase)

            return jnp.sum(jnp.abs(strain) ** 2)

        # Compute gradients
        grad_fn = jax.grad(compute_strain_norm)
        input_params = jnp.array([[0.15, 0.3, -0.2]])

        grads = grad_fn(input_params, params)

        assert grads.shape == input_params.shape
        assert jnp.all(jnp.isfinite(grads))
        assert jnp.any(grads != 0)  # Non-zero gradients


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
