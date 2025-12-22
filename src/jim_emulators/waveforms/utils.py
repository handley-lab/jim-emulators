"""
Utility functions for waveform analysis.

Provides frequency grid generation, inner products, and match calculations.
Uses JAX for GPU acceleration and autodiff compatibility.
"""

from typing import Tuple, Optional
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid

# Enable 64-bit precision for waveform calculations
jax.config.update("jax_enable_x64", True)


def get_frequency_array(
    sampling_frequency: float,
    duration: float,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> jnp.ndarray:
    """
    Generate a frequency array for FFT-based analysis.

    This creates the frequency grid that would result from an FFT of
    time-domain data with the given sampling frequency and duration.

    Parameters
    ----------
    sampling_frequency : float
        Sampling frequency in Hz (e.g., 4096 Hz).
    duration : float
        Duration of the signal in seconds.
    f_min : float, optional
        Minimum frequency to include. If None, starts from 0.
    f_max : float, optional
        Maximum frequency to include. If None, goes to Nyquist.

    Returns
    -------
    frequencies : jnp.ndarray
        Frequency array in Hz.

    Examples
    --------
    >>> freqs = get_frequency_array(4096.0, 16.0, f_min=20.0, f_max=1024.0)
    >>> print(f"Delta f = {freqs[1] - freqs[0]:.4f} Hz")
    Delta f = 0.0625 Hz
    """
    delta_t = 1.0 / sampling_frequency
    n_samples = int(round(duration * sampling_frequency))
    frequencies = jnp.fft.rfftfreq(n_samples, delta_t)

    if f_min is not None or f_max is not None:
        f_min = f_min if f_min is not None else 0.0
        f_max = f_max if f_max is not None else frequencies[-1]
        mask = (frequencies >= f_min) & (frequencies <= f_max)
        frequencies = frequencies[mask]

    return frequencies


def get_geometric_frequency_grid(
    n_points: int = 1000,
    Mf_min: float = 0.003,
    Mf_max: float = 0.25,
) -> jnp.ndarray:
    """
    Generate a log-spaced grid in geometric frequency Mf.

    Geometric frequency Mf = M * f * (G/c^3) is the natural variable
    for waveform emulation since it makes waveforms mass-independent.

    Parameters
    ----------
    n_points : int
        Number of grid points.
    Mf_min : float
        Minimum geometric frequency.
    Mf_max : float
        Maximum geometric frequency.

    Returns
    -------
    Mf_grid : jnp.ndarray
        Log-spaced geometric frequency grid.
    """
    return jnp.logspace(jnp.log10(Mf_min), jnp.log10(Mf_max), n_points)


def geometric_to_physical_frequency(
    Mf: jnp.ndarray,
    total_mass_msun: float,
) -> jnp.ndarray:
    """
    Convert geometric frequency Mf to physical frequency f.

    Parameters
    ----------
    Mf : jnp.ndarray
        Geometric frequency (dimensionless).
    total_mass_msun : float
        Total mass in solar masses.

    Returns
    -------
    f : jnp.ndarray
        Physical frequency in Hz.
    """
    # MTSUN_SI = G * Msun / c^3 in seconds
    MTSUN_SI = 4.925491025543576e-6
    M_seconds = total_mass_msun * MTSUN_SI
    return Mf / M_seconds


def physical_to_geometric_frequency(
    f: jnp.ndarray,
    total_mass_msun: float,
) -> jnp.ndarray:
    """
    Convert physical frequency f to geometric frequency Mf.

    Parameters
    ----------
    f : jnp.ndarray
        Physical frequency in Hz.
    total_mass_msun : float
        Total mass in solar masses.

    Returns
    -------
    Mf : jnp.ndarray
        Geometric frequency (dimensionless).
    """
    MTSUN_SI = 4.925491025543576e-6
    M_seconds = total_mass_msun * MTSUN_SI
    return f * M_seconds


def noise_weighted_inner_product(
    h1: jnp.ndarray,
    h2: jnp.ndarray,
    psd: jnp.ndarray,
    frequencies: jnp.ndarray,
) -> float:
    """
    Compute the noise-weighted inner product between two waveforms.

    The inner product is defined as:
        <h1|h2> = 4 * Re[integral(conj(h1) * h2 / Sn df)]

    Parameters
    ----------
    h1, h2 : jnp.ndarray
        Complex frequency-domain waveforms.
    psd : jnp.ndarray
        Power spectral density of the noise.
    frequencies : jnp.ndarray
        Frequency array in Hz.

    Returns
    -------
    inner_product : float
        The noise-weighted inner product.
    """
    integrand = jnp.conj(h1) * h2 / psd
    return 4.0 * trapezoid(integrand, x=frequencies, axis=-1).real


def compute_match(
    h1: jnp.ndarray,
    h2: jnp.ndarray,
    psd: jnp.ndarray,
    frequencies: jnp.ndarray,
) -> float:
    """
    Compute the match (normalized inner product) between two waveforms.

    The match is defined as:
        match = <h1|h2> / sqrt(<h1|h1> * <h2|h2>)

    This gives a value between 0 and 1, where 1 indicates identical
    waveforms (up to overall amplitude).

    Note: This does not optimize over time/phase shifts. For the
    faithfulness (maximized over shifts), use compute_faithfulness.

    Parameters
    ----------
    h1, h2 : jnp.ndarray
        Complex frequency-domain waveforms.
    psd : jnp.ndarray
        Power spectral density of the noise.
    frequencies : jnp.ndarray
        Frequency array in Hz.

    Returns
    -------
    match : float
        Match between the waveforms (0 to 1).
    """
    h1_h1 = noise_weighted_inner_product(h1, h1, psd, frequencies)
    h2_h2 = noise_weighted_inner_product(h2, h2, psd, frequencies)
    h1_h2 = noise_weighted_inner_product(h1, h2, psd, frequencies)

    return h1_h2 / jnp.sqrt(h1_h1 * h2_h2)


def compute_mismatch(
    h1: jnp.ndarray,
    h2: jnp.ndarray,
    psd: jnp.ndarray,
    frequencies: jnp.ndarray,
) -> float:
    """
    Compute the mismatch between two waveforms.

    Mismatch = 1 - match

    A mismatch of 0 means identical waveforms, 1 means orthogonal.

    Parameters
    ----------
    h1, h2 : jnp.ndarray
        Complex frequency-domain waveforms.
    psd : jnp.ndarray
        Power spectral density of the noise.
    frequencies : jnp.ndarray
        Frequency array in Hz.

    Returns
    -------
    mismatch : float
        Mismatch between the waveforms (0 to 1).
    """
    return 1.0 - compute_match(h1, h2, psd, frequencies)


def load_psd(
    psd_file: str,
    frequencies: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load a PSD from file and optionally interpolate to target frequencies.

    Parameters
    ----------
    psd_file : str
        Path to PSD file. Expected format: columns of (frequency, psd, ...).
    frequencies : jnp.ndarray, optional
        Target frequencies for interpolation.

    Returns
    -------
    psd_frequencies : jnp.ndarray
        Frequency array from file (or target frequencies if provided).
    psd_values : jnp.ndarray
        PSD values (interpolated if frequencies provided).
    """
    data = np.loadtxt(psd_file, unpack=True)
    psd_freqs = jnp.array(data[0])
    psd_vals = jnp.array(data[1])  # Take first PSD column

    if frequencies is not None:
        psd_vals = jnp.interp(frequencies, psd_freqs, psd_vals)
        return frequencies, psd_vals

    return psd_freqs, psd_vals


def compute_snr(
    h: jnp.ndarray,
    psd: jnp.ndarray,
    frequencies: jnp.ndarray,
) -> float:
    """
    Compute the optimal signal-to-noise ratio (SNR) of a waveform.

    SNR = sqrt(<h|h>)

    Parameters
    ----------
    h : jnp.ndarray
        Complex frequency-domain waveform.
    psd : jnp.ndarray
        Power spectral density of the noise.
    frequencies : jnp.ndarray
        Frequency array in Hz.

    Returns
    -------
    snr : float
        Optimal SNR.
    """
    h_h = noise_weighted_inner_product(h, h, psd, frequencies)
    return jnp.sqrt(h_h)


# =============================================================================
# Time Domain Utilities
# =============================================================================

def fd_to_td(
    h_fd: jnp.ndarray,
    delta_f: float,
    center: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert frequency-domain waveform to time domain via inverse FFT.

    Parameters
    ----------
    h_fd : jnp.ndarray
        Complex frequency-domain waveform (positive frequencies only,
        as returned by rfft conventions).
    delta_f : float
        Frequency spacing in Hz.
    center : bool
        If True, shift waveform so peak amplitude is at center of window.

    Returns
    -------
    times : jnp.ndarray
        Time array in seconds.
    h_td : jnp.ndarray
        Real time-domain waveform.

    Notes
    -----
    The frequency array is assumed to start at f=0 and have spacing delta_f.
    The resulting time array has spacing dt = 1/(N*delta_f) where N is the
    number of time samples (approximately 2x the frequency samples for rfft).
    """
    # Number of frequency points (positive frequencies including 0 and Nyquist)
    n_freq = len(h_fd)

    # Number of time samples (for irfft, n_time = 2*(n_freq-1) if even)
    n_time = 2 * (n_freq - 1)

    # Time spacing and array
    duration = 1.0 / delta_f
    delta_t = duration / n_time
    times = jnp.arange(n_time) * delta_t

    # Inverse FFT (irfft expects positive frequencies, returns real signal)
    # Scale by delta_f to get correct amplitude (FFT normalization)
    h_td = jnp.fft.irfft(h_fd) * delta_f * n_time

    if center:
        # Find peak of absolute value and shift to center
        peak_idx = jnp.argmax(jnp.abs(h_td))
        center_idx = n_time // 2
        shift = center_idx - peak_idx
        h_td = jnp.roll(h_td, shift)
        # Adjust time array to be centered on zero
        times = times - times[center_idx]

    return times, h_td


def fd_to_td_centered_at_merger(
    h_fd: jnp.ndarray,
    frequencies: jnp.ndarray,
    t_center: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert to time domain with merger at specified time.

    This uses the frequency-domain phase shift property:
        h(t - t0) <-> h(f) * exp(-2*pi*i*f*t0)

    Parameters
    ----------
    h_fd : jnp.ndarray
        Complex frequency-domain waveform.
    frequencies : jnp.ndarray
        Frequency array in Hz.
    t_center : float
        Time at which to place the merger (peak amplitude).

    Returns
    -------
    times : jnp.ndarray
        Time array centered on t_center.
    h_td : jnp.ndarray
        Time-domain waveform with peak at t_center.
    """
    delta_f = frequencies[1] - frequencies[0]
    n_freq = len(h_fd)
    n_time = 2 * (n_freq - 1)
    duration = 1.0 / delta_f
    delta_t = duration / n_time

    # First transform without centering to find peak location
    h_td_raw = jnp.fft.irfft(h_fd) * delta_f * n_time
    peak_idx = jnp.argmax(jnp.abs(h_td_raw))
    t_peak = peak_idx * delta_t

    # Time shift needed
    t_shift = t_center - t_peak + duration / 2  # shift peak to center, then to t_center

    # Apply phase shift in frequency domain
    phase_shift = jnp.exp(-2j * jnp.pi * frequencies * t_shift)
    h_fd_shifted = h_fd * phase_shift

    # Transform
    h_td = jnp.fft.irfft(h_fd_shifted) * delta_f * n_time

    # Time array centered on t_center
    times = jnp.arange(n_time) * delta_t - duration / 2 + t_center

    return times, h_td


def geometric_fd_to_td(
    h_Mf: jnp.ndarray,
    Mf_grid: jnp.ndarray,
    center: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert geometric frequency-domain waveform to geometric time domain.

    This transforms h(Mf) -> h(t/M), giving a mass-independent time-domain
    representation.

    Parameters
    ----------
    h_Mf : jnp.ndarray
        Complex waveform in geometric frequency Mf.
    Mf_grid : jnp.ndarray
        Geometric frequency grid (must be uniformly spaced for FFT).
    center : bool
        If True, center the peak at t/M = 0.

    Returns
    -------
    tM_grid : jnp.ndarray
        Geometric time array t/M (dimensionless).
    h_tM : jnp.ndarray
        Time-domain waveform in geometric units.

    Notes
    -----
    The geometric time t/M has units of total mass. To convert to seconds:
        t_seconds = (t/M) * M_total * MTSUN_SI

    where MTSUN_SI = G*Msun/c^3 ≈ 4.926e-6 seconds.
    """
    # Check for uniform spacing
    delta_Mf = Mf_grid[1] - Mf_grid[0]

    n_freq = len(h_Mf)
    n_time = 2 * (n_freq - 1)

    # Geometric time spacing: d(t/M) = 1 / (N * d(Mf))
    duration_tM = 1.0 / delta_Mf
    delta_tM = duration_tM / n_time
    tM_grid = jnp.arange(n_time) * delta_tM

    # Inverse FFT
    h_tM = jnp.fft.irfft(h_Mf) * delta_Mf * n_time

    if center:
        # Find peak and shift to center
        peak_idx = jnp.argmax(jnp.abs(h_tM))
        center_idx = n_time // 2
        shift = center_idx - peak_idx
        h_tM = jnp.roll(h_tM, shift)
        # Center time array on zero
        tM_grid = tM_grid - tM_grid[center_idx]

    return tM_grid, h_tM


def geometric_time_to_physical(
    tM: jnp.ndarray,
    total_mass_msun: float,
) -> jnp.ndarray:
    """
    Convert geometric time t/M to physical time in seconds.

    Parameters
    ----------
    tM : jnp.ndarray
        Geometric time (dimensionless, in units of total mass).
    total_mass_msun : float
        Total mass in solar masses.

    Returns
    -------
    t : jnp.ndarray
        Physical time in seconds.
    """
    MTSUN_SI = 4.925491025543576e-6  # G*Msun/c^3 in seconds
    M_seconds = total_mass_msun * MTSUN_SI
    return tM * M_seconds


def physical_time_to_geometric(
    t: jnp.ndarray,
    total_mass_msun: float,
) -> jnp.ndarray:
    """
    Convert physical time in seconds to geometric time t/M.

    Parameters
    ----------
    t : jnp.ndarray
        Physical time in seconds.
    total_mass_msun : float
        Total mass in solar masses.

    Returns
    -------
    tM : jnp.ndarray
        Geometric time (dimensionless).
    """
    MTSUN_SI = 4.925491025543576e-6
    M_seconds = total_mass_msun * MTSUN_SI
    return t / M_seconds


def interpolate_to_uniform_grid(
    data: jnp.ndarray,
    grid: jnp.ndarray,
    n_points: Optional[int] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Interpolate data onto a uniform grid (required for FFT).

    Parameters
    ----------
    data : jnp.ndarray
        Data values (can be complex).
    grid : jnp.ndarray
        Original grid points (may be non-uniform, e.g., log-spaced).
    n_points : int, optional
        Number of points in uniform grid. Defaults to len(grid).

    Returns
    -------
    uniform_grid : jnp.ndarray
        Uniformly spaced grid.
    uniform_data : jnp.ndarray
        Data interpolated onto uniform grid.
    """
    if n_points is None:
        n_points = len(grid)

    uniform_grid = jnp.linspace(grid[0], grid[-1], n_points)

    # Interpolate real and imaginary parts separately if complex
    if jnp.iscomplexobj(data):
        real_interp = jnp.interp(uniform_grid, grid, data.real)
        imag_interp = jnp.interp(uniform_grid, grid, data.imag)
        uniform_data = real_interp + 1j * imag_interp
    else:
        uniform_data = jnp.interp(uniform_grid, grid, data)

    return uniform_grid, uniform_data
