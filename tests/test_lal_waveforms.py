"""
Tests for LAL waveform wrapper.

Run with: pytest tests/test_lal_waveforms.py -v
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, "src")

from jim_emulators.waveforms.lal_waveforms import (
    WaveformParameters,
    generate_fd_waveform,
    generate_fd_waveform_on_grid,
    generate_fd_mode,
    generate_fd_modes,
    get_waveform_amplitude_phase,
    SUPPORTED_APPROXIMANTS,
)
from jim_emulators.waveforms.utils import (
    get_frequency_array,
    compute_match,
    load_psd,
    compute_snr,
)


class TestWaveformParameters:
    """Tests for WaveformParameters dataclass."""

    def test_basic_creation(self):
        """Test basic parameter creation."""
        params = WaveformParameters(mass_1=30.0, mass_2=25.0)
        assert params.mass_1 == 30.0
        assert params.mass_2 == 25.0
        assert params.approximant == "IMRPhenomXPHM"

    def test_mass_swap(self):
        """Test that masses are swapped so m1 >= m2."""
        params = WaveformParameters(mass_1=10.0, mass_2=30.0)
        assert params.mass_1 == 30.0
        assert params.mass_2 == 10.0

    def test_spin_swap_with_mass(self):
        """Test that spins are swapped with masses."""
        params = WaveformParameters(
            mass_1=10.0, mass_2=30.0,
            chi1z=0.5, chi2z=-0.3
        )
        assert params.mass_1 == 30.0
        assert params.chi1z == -0.3  # Swapped
        assert params.chi2z == 0.5   # Swapped

    def test_default_f_ref(self):
        """Test that f_ref defaults to f_min."""
        params = WaveformParameters(mass_1=30.0, mass_2=25.0, f_min=15.0)
        assert params.f_ref == 15.0

    def test_derived_quantities(self):
        """Test derived mass quantities."""
        params = WaveformParameters(mass_1=30.0, mass_2=30.0)
        assert params.total_mass == 60.0
        assert params.mass_ratio == 1.0
        assert params.symmetric_mass_ratio == 0.25
        assert np.isclose(params.chirp_mass, 60.0 * 0.25**0.6)

    def test_chi_eff(self):
        """Test effective spin calculation."""
        params = WaveformParameters(
            mass_1=30.0, mass_2=30.0,
            chi1z=0.5, chi2z=0.5
        )
        assert params.chi_eff == 0.5

    def test_invalid_approximant(self):
        """Test that invalid approximant raises error."""
        with pytest.raises(ValueError, match="not supported"):
            WaveformParameters(mass_1=30.0, mass_2=25.0, approximant="InvalidWF")


class TestWaveformGeneration:
    """Tests for waveform generation."""

    @pytest.fixture
    def default_params(self):
        """Default parameters for testing."""
        return WaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            chi1z=0.3,
            chi2z=-0.2,
            luminosity_distance=400.0,
            f_min=20.0,
            f_max=512.0,
            delta_f=0.25,
        )

    def test_waveform_generation(self, default_params):
        """Test basic waveform generation."""
        freqs, hp, hc = generate_fd_waveform(default_params)

        # Check shapes
        assert len(freqs) == len(hp) == len(hc)
        assert len(freqs) > 0

        # Check frequency array starts from 0
        assert freqs[0] == 0.0

        # Check frequency spacing
        df = freqs[1] - freqs[0]
        assert np.isclose(df, default_params.delta_f)

        # Check waveform is complex
        assert hp.dtype == np.complex128
        assert hc.dtype == np.complex128

    def test_waveform_is_nonzero_in_band(self, default_params):
        """Test that waveform is nonzero in the frequency band."""
        freqs, hp, hc = generate_fd_waveform(default_params)

        # Get mask for frequency band
        mask = (freqs >= default_params.f_min) & (freqs <= default_params.f_max)

        # Waveform should be nonzero in band
        assert np.any(np.abs(hp[mask]) > 0)
        assert np.any(np.abs(hc[mask]) > 0)

    def test_distance_scaling(self):
        """Test that amplitude scales inversely with distance."""
        params1 = WaveformParameters(
            mass_1=30.0, mass_2=25.0,
            luminosity_distance=100.0,
            f_min=20.0, f_max=256.0, delta_f=0.5
        )
        params2 = WaveformParameters(
            mass_1=30.0, mass_2=25.0,
            luminosity_distance=200.0,
            f_min=20.0, f_max=256.0, delta_f=0.5
        )

        _, hp1, _ = generate_fd_waveform(params1)
        _, hp2, _ = generate_fd_waveform(params2)

        # Find a frequency where both are nonzero
        nonzero = (np.abs(hp1) > 0) & (np.abs(hp2) > 0)
        if np.any(nonzero):
            idx = np.where(nonzero)[0][len(np.where(nonzero)[0])//2]
            ratio = np.abs(hp1[idx]) / np.abs(hp2[idx])
            assert np.isclose(ratio, 2.0, rtol=1e-3)

    def test_all_approximants(self):
        """Test that all supported approximants work."""
        for approx_name in SUPPORTED_APPROXIMANTS.keys():
            params = WaveformParameters(
                mass_1=30.0, mass_2=25.0,
                f_min=20.0, f_max=256.0, delta_f=1.0,
                approximant=approx_name
            )
            freqs, hp, hc = generate_fd_waveform(params)
            assert len(freqs) > 0
            assert np.any(np.abs(hp) > 0)


class TestAmplitudePhase:
    """Tests for amplitude/phase extraction."""

    def test_amplitude_phase_extraction(self):
        """Test amplitude and phase extraction."""
        params = WaveformParameters(
            mass_1=30.0, mass_2=25.0,
            f_min=20.0, f_max=256.0, delta_f=0.5
        )
        freqs, hp, hc = generate_fd_waveform(params)
        amp_p, phase_p, amp_c, phase_c = get_waveform_amplitude_phase(freqs, hp, hc)

        # Amplitudes should be non-negative
        assert np.all(amp_p >= 0)
        assert np.all(amp_c >= 0)

        # Reconstruct from amp/phase should match original
        hp_reconstructed = amp_p * np.exp(1j * phase_p)
        np.testing.assert_allclose(hp_reconstructed, hp, rtol=1e-10)


class TestModeExtraction:
    """Tests for individual mode extraction."""

    @pytest.fixture
    def default_params(self):
        """Default parameters for mode testing."""
        return WaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            chi1z=0.3,
            chi2z=-0.2,
            luminosity_distance=100.0,
            f_min=20.0,
            f_max=512.0,
            delta_f=0.5,
        )

    def test_generate_22_mode(self, default_params):
        """Test generation of (2,2) mode."""
        freqs, h22 = generate_fd_mode(default_params, ell=2, emm=2)

        # Check output shapes
        assert len(freqs) == len(h22)
        assert len(freqs) > 0

        # Check waveform is complex
        assert h22.dtype == np.complex128

        # Check mode is nonzero in band
        mask = (freqs >= default_params.f_min) & (freqs <= default_params.f_max)
        assert np.any(np.abs(h22[mask]) > 0)

    def test_generate_higher_modes(self, default_params):
        """Test generation of higher modes."""
        for ell, emm in [(2, 1), (3, 3), (3, 2), (4, 4)]:
            freqs, hlm = generate_fd_mode(default_params, ell=ell, emm=emm)
            assert len(freqs) == len(hlm)
            # Higher modes may be weaker but should still exist
            assert np.any(np.abs(hlm) > 0)

    def test_generate_multiple_modes(self, default_params):
        """Test generation of multiple modes at once."""
        modes_to_generate = [(2, 2), (2, 1), (3, 3)]
        freqs, mode_dict = generate_fd_modes(default_params, modes=modes_to_generate)

        # Check all requested modes are present
        assert set(mode_dict.keys()) == set(modes_to_generate)

        # Check all modes have same length as frequency array
        for mode, hlm in mode_dict.items():
            assert len(hlm) == len(freqs)

    def test_mode_distance_scaling(self, default_params):
        """Test that mode amplitude scales inversely with distance."""
        params1 = WaveformParameters(
            mass_1=30.0, mass_2=25.0,
            luminosity_distance=100.0,
            f_min=20.0, f_max=256.0, delta_f=0.5
        )
        params2 = WaveformParameters(
            mass_1=30.0, mass_2=25.0,
            luminosity_distance=200.0,
            f_min=20.0, f_max=256.0, delta_f=0.5
        )

        freqs1, h22_1 = generate_fd_mode(params1, ell=2, emm=2)
        freqs2, h22_2 = generate_fd_mode(params2, ell=2, emm=2)

        # Find a frequency where both are nonzero
        nonzero = (np.abs(h22_1) > 0) & (np.abs(h22_2) > 0)
        if np.any(nonzero):
            idx = np.where(nonzero)[0][len(np.where(nonzero)[0])//2]
            ratio = np.abs(h22_1[idx]) / np.abs(h22_2[idx])
            assert np.isclose(ratio, 2.0, rtol=1e-3)

    def test_mode_amplitude_phase(self, default_params):
        """Test extraction of amplitude and phase from mode."""
        freqs, h22 = generate_fd_mode(default_params, ell=2, emm=2)

        # Get valid data
        mask = np.abs(h22) > 0
        h22_valid = h22[mask]

        # Extract amplitude and phase
        amplitude = np.abs(h22_valid)
        phase = np.unwrap(np.angle(h22_valid))

        # Amplitudes should be positive
        assert np.all(amplitude > 0)

        # Phase should be continuous (no jumps > pi after unwrapping)
        phase_diff = np.abs(np.diff(phase))
        assert np.all(phase_diff < np.pi)

        # Reconstruct should match original
        h22_reconstructed = amplitude * np.exp(1j * phase)
        np.testing.assert_allclose(h22_reconstructed, h22_valid, rtol=1e-10)


class TestMatch:
    """Tests for match calculations."""

    def test_self_match(self):
        """Test that a waveform has match 1 with itself."""
        import jax.numpy as jnp

        params = WaveformParameters(
            mass_1=30.0, mass_2=25.0,
            f_min=20.0, f_max=256.0, delta_f=0.5
        )
        freqs, hp, _ = generate_fd_waveform(params)

        # Create simple flat PSD
        mask = (freqs >= params.f_min) & (freqs <= params.f_max)
        freqs_band = jnp.array(freqs[mask])
        hp_band = jnp.array(hp[mask])
        psd = jnp.ones_like(freqs_band) * 1e-46

        match = compute_match(hp_band, hp_band, psd, freqs_band)
        assert np.isclose(match, 1.0, rtol=1e-6)

    def test_orthogonal_waveforms(self):
        """Test that orthogonal waveforms have match near 0."""
        import jax.numpy as jnp

        # Create two very different waveforms
        params1 = WaveformParameters(
            mass_1=10.0, mass_2=10.0,
            f_min=20.0, f_max=512.0, delta_f=0.5
        )
        params2 = WaveformParameters(
            mass_1=80.0, mass_2=80.0,
            f_min=20.0, f_max=512.0, delta_f=0.5
        )

        freqs1, hp1, _ = generate_fd_waveform(params1)
        freqs2, hp2, _ = generate_fd_waveform(params2)

        # Use common frequency range
        f_min, f_max = 50.0, 200.0
        mask1 = (freqs1 >= f_min) & (freqs1 <= f_max)
        mask2 = (freqs2 >= f_min) & (freqs2 <= f_max)

        freqs_band = jnp.array(freqs1[mask1])
        hp1_band = jnp.array(hp1[mask1])
        hp2_band = jnp.array(hp2[mask2][:len(hp1_band)])  # Same length
        psd = jnp.ones_like(freqs_band) * 1e-46

        match = compute_match(hp1_band, hp2_band, psd, freqs_band)
        # Very different masses should give low match
        assert match < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
