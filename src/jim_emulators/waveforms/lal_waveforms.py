"""
LAL waveform generation wrapper.

Provides a clean, testable interface to LALSimulation frequency-domain waveforms.
All physical quantities use SI units internally, with convenient input in
astrophysical units (solar masses, Mpc).

Mode Selection
--------------
Higher-mode waveforms (XPHM, XHM) support mode selection via the `mode_array`
parameter. This allows generating waveforms with only specific (l, m) modes.

Available modes for IMRPhenomXPHM/XHM:
    (2, 2), (2, -2)  - Dominant quadrupole
    (2, 1), (2, -1)  - Subdominant quadrupole
    (3, 3), (3, -3)  - Octupole
    (3, 2), (3, -2)  - Mixed
    (4, 4), (4, -4)  - Hexadecapole

Note: XAS only contains the (2, ±2) mode by construction.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

import lal
import lalsimulation as lalsim

# Physical constants from LAL
MSUN_SI = lal.MSUN_SI  # Solar mass in kg
MTSUN_SI = lal.MTSUN_SI  # Solar mass in seconds (G*Msun/c^3)
PC_SI = lal.PC_SI  # Parsec in meters
MPC_SI = PC_SI * 1e6  # Megaparsec in meters

# Supported approximants
SUPPORTED_APPROXIMANTS = {
    "IMRPhenomXPHM": lalsim.IMRPhenomXPHM,
    "IMRPhenomXHM": lalsim.IMRPhenomXHM,
    "IMRPhenomXAS": lalsim.IMRPhenomXAS,
    "IMRPhenomD": lalsim.IMRPhenomD,
}

# Default modes for higher-mode approximants
DEFAULT_MODES_XPHM = [(2, 2), (2, -2), (2, 1), (2, -1), (3, 3), (3, -3), (3, 2), (3, -2), (4, 4), (4, -4)]
DEFAULT_MODES_XHM = [(2, 2), (2, -2), (2, 1), (2, -1), (3, 3), (3, -3), (3, 2), (3, -2), (4, 4), (4, -4)]
DEFAULT_MODES_XAS = [(2, 2), (2, -2)]  # XAS only has dominant mode


def create_mode_array(modes: List[Tuple[int, int]]) -> "lal.ModeArray":
    """
    Create a LAL ModeArray from a list of (l, m) tuples.

    Parameters
    ----------
    modes : list of (l, m) tuples
        Spherical harmonic modes to include.
        Example: [(2, 2), (2, -2)] for dominant quadrupole only.

    Returns
    -------
    lal_mode_array : LAL ModeArray object
        Mode array for use with SimInspiralWaveformParamsInsertModeArray.
    """
    lal_mode_array = lalsim.SimInspiralCreateModeArray()
    for l, m in modes:
        lalsim.SimInspiralModeArrayActivateMode(lal_mode_array, l, m)
    return lal_mode_array


def get_available_modes(approximant: str) -> List[Tuple[int, int]]:
    """
    Get the list of available modes for a given approximant.

    Parameters
    ----------
    approximant : str
        Waveform approximant name.

    Returns
    -------
    modes : list of (l, m) tuples
        Available spherical harmonic modes.
    """
    if approximant == "IMRPhenomXPHM":
        return DEFAULT_MODES_XPHM.copy()
    elif approximant == "IMRPhenomXHM":
        return DEFAULT_MODES_XHM.copy()
    elif approximant in ("IMRPhenomXAS", "IMRPhenomD"):
        return DEFAULT_MODES_XAS.copy()
    else:
        raise ValueError(f"Unknown approximant: {approximant}")


@dataclass
class WaveformParameters:
    """
    Parameters for frequency-domain waveform generation.

    All masses in solar masses, distance in Mpc, angles in radians.
    Spins are dimensionless (|chi| <= 1).

    Attributes
    ----------
    mass_1 : float
        Mass of the primary (heavier) object in solar masses.
    mass_2 : float
        Mass of the secondary (lighter) object in solar masses.
    chi1x, chi1y, chi1z : float
        Dimensionless spin components of primary.
    chi2x, chi2y, chi2z : float
        Dimensionless spin components of secondary.
    luminosity_distance : float
        Luminosity distance in Mpc.
    inclination : float
        Inclination angle in radians (angle between orbital angular momentum and line of sight).
    phase : float
        Reference phase in radians.
    f_min : float
        Minimum frequency in Hz.
    f_max : float
        Maximum frequency in Hz.
    delta_f : float
        Frequency spacing in Hz.
    f_ref : float
        Reference frequency in Hz (typically set to f_min).
    approximant : str
        Waveform approximant name (e.g., "IMRPhenomXPHM").
    """
    mass_1: float
    mass_2: float
    chi1x: float = 0.0
    chi1y: float = 0.0
    chi1z: float = 0.0
    chi2x: float = 0.0
    chi2y: float = 0.0
    chi2z: float = 0.0
    luminosity_distance: float = 1.0  # Mpc, default to 1 for easy scaling
    inclination: float = 0.0
    phase: float = 0.0
    f_min: float = 20.0
    f_max: float = 1024.0
    delta_f: float = 0.125
    f_ref: Optional[float] = None  # Defaults to f_min if None
    approximant: str = "IMRPhenomXPHM"

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Ensure m1 >= m2 (swap if necessary)
        if self.mass_1 < self.mass_2:
            self.mass_1, self.mass_2 = self.mass_2, self.mass_1
            # Also swap spins
            self.chi1x, self.chi2x = self.chi2x, self.chi1x
            self.chi1y, self.chi2y = self.chi2y, self.chi1y
            self.chi1z, self.chi2z = self.chi2z, self.chi1z

        # Set reference frequency to f_min if not specified
        if self.f_ref is None:
            self.f_ref = self.f_min

        # Validate approximant
        if self.approximant not in SUPPORTED_APPROXIMANTS:
            raise ValueError(
                f"Approximant '{self.approximant}' not supported. "
                f"Choose from: {list(SUPPORTED_APPROXIMANTS.keys())}"
            )

    @property
    def total_mass(self) -> float:
        """Total mass in solar masses."""
        return self.mass_1 + self.mass_2

    @property
    def chirp_mass(self) -> float:
        """Chirp mass in solar masses."""
        return (self.mass_1 * self.mass_2) ** 0.6 / self.total_mass ** 0.2

    @property
    def symmetric_mass_ratio(self) -> float:
        """Symmetric mass ratio eta = m1*m2/(m1+m2)^2."""
        return self.mass_1 * self.mass_2 / self.total_mass ** 2

    @property
    def mass_ratio(self) -> float:
        """Mass ratio q = m2/m1 <= 1."""
        return self.mass_2 / self.mass_1

    @property
    def chi_eff(self) -> float:
        """Effective aligned spin parameter."""
        return (self.mass_1 * self.chi1z + self.mass_2 * self.chi2z) / self.total_mass

    def to_lal_units(self) -> Dict[str, float]:
        """Convert to LAL SI units."""
        return {
            "mass_1_si": self.mass_1 * MSUN_SI,
            "mass_2_si": self.mass_2 * MSUN_SI,
            "distance_si": self.luminosity_distance * MPC_SI,
        }


def generate_fd_waveform(
    params: WaveformParameters,
    mode_array: Optional[List[Tuple[int, int]]] = None,
    disable_multibanding: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a frequency-domain waveform using LALSimulation.

    Parameters
    ----------
    params : WaveformParameters
        Waveform parameters dataclass.
    mode_array : list of (l, m) tuples, optional
        Spherical harmonic modes to include. If None, uses all available modes
        for the approximant. Only relevant for higher-mode approximants (XPHM, XHM).
        Example: [(2, 2), (2, -2)] for dominant quadrupole only.
    disable_multibanding : bool, optional
        If True, disables LAL's multibanding optimization for maximum numerical
        precision. This is slower but produces smoother waveforms, especially
        at low amplitudes. Default is False (use LAL defaults).

    Returns
    -------
    frequencies : np.ndarray
        Frequency array in Hz.
    hp : np.ndarray
        Plus polarization (complex).
    hc : np.ndarray
        Cross polarization (complex).

    Examples
    --------
    >>> params = WaveformParameters(
    ...     mass_1=30.0, mass_2=25.0,
    ...     chi1z=0.3, chi2z=-0.2,
    ...     luminosity_distance=400.0,
    ...     f_min=20.0, f_max=1024.0, delta_f=0.125
    ... )
    >>> freqs, hp, hc = generate_fd_waveform(params)

    # Generate with only (2,2) mode:
    >>> freqs, hp, hc = generate_fd_waveform(params, mode_array=[(2, 2), (2, -2)])

    # Generate with maximum precision (no multibanding):
    >>> freqs, hp, hc = generate_fd_waveform(params, disable_multibanding=True)
    """
    # Convert to SI units
    mass_1_si = params.mass_1 * MSUN_SI
    mass_2_si = params.mass_2 * MSUN_SI
    distance_si = params.luminosity_distance * MPC_SI

    # Get approximant enum
    approximant = SUPPORTED_APPROXIMANTS[params.approximant]

    # Create LAL dictionary for extra parameters
    laldict = None
    if mode_array is not None or disable_multibanding:
        laldict = lal.CreateDict()

        if mode_array is not None:
            lal_mode_array = create_mode_array(mode_array)
            lalsim.SimInspiralWaveformParamsInsertModeArray(laldict, lal_mode_array)

        if disable_multibanding:
            # Setting threshold to 0 disables multibanding entirely
            lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(laldict, 0.0)
            lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(laldict, 0.0)

    # Call LALSimulation
    hp_lal, hc_lal = lalsim.SimInspiralChooseFDWaveform(
        mass_1_si,
        mass_2_si,
        float(params.chi1x),
        float(params.chi1y),
        float(params.chi1z),
        float(params.chi2x),
        float(params.chi2y),
        float(params.chi2z),
        distance_si,
        float(params.inclination),
        float(params.phase),
        0.0,  # longitude of ascending nodes
        0.0,  # eccentricity
        0.0,  # mean anomaly
        float(params.delta_f),
        float(params.f_min),
        float(params.f_max),
        float(params.f_ref),
        laldict,  # LAL dictionary for extra parameters (mode selection)
        approximant,
    )

    # Extract data from LAL structures
    hp_data = hp_lal.data.data
    hc_data = hc_lal.data.data

    # Build frequency array (LAL returns data starting from f=0)
    n_points = len(hp_data)
    frequencies = np.arange(n_points) * params.delta_f

    return frequencies, hp_data, hc_data


def generate_fd_waveform_on_grid(
    params: WaveformParameters,
    target_frequencies: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate waveform and interpolate onto a target frequency grid.

    This is useful when you need waveforms on a specific grid (e.g., for
    comparison or training data generation).

    Parameters
    ----------
    params : WaveformParameters
        Waveform parameters.
    target_frequencies : np.ndarray
        Target frequency array in Hz.

    Returns
    -------
    hp : np.ndarray
        Plus polarization on target grid (complex).
    hc : np.ndarray
        Cross polarization on target grid (complex).
    """
    # Generate waveform
    freqs, hp, hc = generate_fd_waveform(params)

    # Find valid frequency range
    f_min_actual = max(params.f_min, target_frequencies[0])
    f_max_actual = min(freqs[-1], target_frequencies[-1])

    # Create mask for valid frequencies
    target_mask = (target_frequencies >= f_min_actual) & (target_frequencies <= f_max_actual)
    source_mask = (freqs >= f_min_actual) & (freqs <= f_max_actual)

    # Interpolate real and imaginary parts separately
    hp_interp = np.zeros(len(target_frequencies), dtype=np.complex128)
    hc_interp = np.zeros(len(target_frequencies), dtype=np.complex128)

    if np.any(source_mask):
        hp_interp[target_mask] = np.interp(
            target_frequencies[target_mask], freqs[source_mask], hp[source_mask]
        )
        hc_interp[target_mask] = np.interp(
            target_frequencies[target_mask], freqs[source_mask], hc[source_mask]
        )

    return hp_interp, hc_interp


def get_waveform_amplitude_phase(
    frequencies: np.ndarray,
    hp: np.ndarray,
    hc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract amplitude and phase from plus/cross polarizations.

    For the dominant (2,2) mode at face-on inclination, hp contains
    the full waveform. For general inclinations, we compute amplitude
    and phase from the complex strain.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array.
    hp, hc : np.ndarray
        Plus and cross polarizations.

    Returns
    -------
    amp_plus : np.ndarray
        Amplitude of plus polarization.
    phase_plus : np.ndarray
        Unwrapped phase of plus polarization.
    amp_cross : np.ndarray
        Amplitude of cross polarization.
    phase_cross : np.ndarray
        Unwrapped phase of cross polarization.
    """
    # Compute amplitude and phase for each polarization
    amp_plus = np.abs(hp)
    amp_cross = np.abs(hc)

    # Unwrap phase to avoid discontinuities
    phase_plus = np.unwrap(np.angle(hp))
    phase_cross = np.unwrap(np.angle(hc))

    return amp_plus, phase_plus, amp_cross, phase_cross


def generate_fd_mode(
    params: WaveformParameters,
    ell: int,
    emm: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single spherical harmonic mode h_lm in the frequency domain.

    This extracts the individual mode rather than the combined polarizations,
    which is the correct representation for neural network emulation. For
    non-precessing (aligned-spin) waveforms, modes with m < 0 can be obtained
    from conjugate symmetry: h_{l,-m}(f) = (-1)^l * h_{lm}^*(f).

    Parameters
    ----------
    params : WaveformParameters
        Waveform parameters. Only aligned spins (chi1z, chi2z) are used;
        transverse spin components are ignored.
    ell : int
        Spherical harmonic degree (l >= 2).
    emm : int
        Spherical harmonic order (-l <= m <= l, typically m > 0).

    Returns
    -------
    frequencies : np.ndarray
        Frequency array in Hz (uniform grid from 0 to f_max with spacing delta_f).
    h_lm : np.ndarray
        Complex mode amplitude h_lm(f).

    Notes
    -----
    Uses LALSimulation's SimIMRPhenomXHMGenerateFDOneMode for aligned-spin
    waveforms. The output is the raw mode h_lm, not weighted by spherical
    harmonics. This function generates on LAL's native uniform frequency grid.

    For evaluation at arbitrary frequencies (e.g., log-spaced), use
    generate_fd_mode_at_frequencies() instead.

    The mode satisfies: h(f) = sum_{lm} h_lm(f) * Y_{lm}(iota, phi)

    Examples
    --------
    >>> params = WaveformParameters(
    ...     mass_1=30.0, mass_2=25.0,
    ...     chi1z=0.3, chi2z=-0.2,
    ...     f_min=20.0, f_max=1024.0, delta_f=0.125
    ... )
    >>> freqs, h22 = generate_fd_mode(params, ell=2, emm=2)
    >>> amplitude = np.abs(h22)
    >>> phase = np.unwrap(np.angle(h22))
    """
    # Convert to SI units
    mass_1_si = params.mass_1 * MSUN_SI
    mass_2_si = params.mass_2 * MSUN_SI
    distance_si = params.luminosity_distance * MPC_SI

    # Reference frequency defaults to f_min
    f_ref = params.f_ref if params.f_ref is not None else params.f_min

    # Generate single mode using XHM (aligned-spin higher modes)
    hlm_lal = lalsim.SimIMRPhenomXHMGenerateFDOneMode(
        mass_1_si,
        mass_2_si,
        float(params.chi1z),
        float(params.chi2z),
        int(ell),
        int(emm),
        distance_si,
        float(params.f_min),
        float(params.f_max),
        float(params.delta_f),
        float(params.phase),
        float(f_ref),
        None,  # LAL dictionary
    )

    # Extract data
    h_lm = hlm_lal.data.data

    # Build frequency array
    n_points = len(h_lm)
    frequencies = np.arange(n_points) * params.delta_f

    return frequencies, h_lm


def generate_fd_mode_at_frequencies(
    frequencies: np.ndarray,
    mass_1: float,
    mass_2: float,
    chi1z: float,
    chi2z: float,
    ell: int,
    emm: int,
    luminosity_distance: float = 1.0,
    phase: float = 0.0,
    f_ref: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a single mode h_lm evaluated at specified frequencies.

    Unlike generate_fd_mode(), this evaluates the waveform at exactly the
    frequencies provided, without interpolation. Use this for log-spaced
    or other non-uniform frequency grids.

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequencies in Hz at which to evaluate the mode.
    mass_1 : float
        Mass of primary in solar masses.
    mass_2 : float
        Mass of secondary in solar masses.
    chi1z, chi2z : float
        Aligned spin components (dimensionless).
    ell, emm : int
        Spherical harmonic mode numbers.
    luminosity_distance : float
        Luminosity distance in Mpc.
    phase : float
        Reference phase in radians.
    f_ref : float, optional
        Reference frequency in Hz. Defaults to first frequency in array.

    Returns
    -------
    h_lm : np.ndarray
        Complex mode amplitude h_lm evaluated at the input frequencies.

    Notes
    -----
    Uses LALSimulation's SimIMRPhenomXHMFrequencySequenceOneMode which
    evaluates the waveform model at arbitrary frequency points without
    interpolation.

    Examples
    --------
    >>> # Log-spaced frequency grid
    >>> freqs = np.logspace(np.log10(20), np.log10(512), 1000)
    >>> h22 = generate_fd_mode_at_frequencies(
    ...     freqs, mass_1=30.0, mass_2=25.0,
    ...     chi1z=0.3, chi2z=-0.2, ell=2, emm=2
    ... )
    """
    # Ensure m1 >= m2
    if mass_1 < mass_2:
        mass_1, mass_2 = mass_2, mass_1
        chi1z, chi2z = chi2z, chi1z

    # Convert to SI units
    mass_1_si = mass_1 * MSUN_SI
    mass_2_si = mass_2 * MSUN_SI
    distance_si = luminosity_distance * MPC_SI

    # Reference frequency defaults to first frequency
    if f_ref is None:
        f_ref = float(frequencies[0])

    # Create LAL frequency vector
    freqs_lal = lal.CreateREAL8Vector(len(frequencies))
    freqs_lal.data[:] = frequencies

    # Evaluate mode at specified frequencies
    hlm_lal = lalsim.SimIMRPhenomXHMFrequencySequenceOneMode(
        freqs_lal,
        mass_1_si,
        mass_2_si,
        float(chi1z),
        float(chi2z),
        int(ell),
        int(emm),
        distance_si,
        float(phase),
        float(f_ref),
        None,  # LAL dictionary
    )

    return hlm_lal.data.data


def generate_fd_modes(
    params: WaveformParameters,
    modes: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    """
    Generate multiple spherical harmonic modes in the frequency domain.

    Parameters
    ----------
    params : WaveformParameters
        Waveform parameters.
    modes : list of (l, m) tuples, optional
        Modes to generate. Default is [(2,2), (2,1), (3,3), (3,2), (4,4)]
        (positive m only, as negative m can be obtained by conjugate symmetry).

    Returns
    -------
    frequencies : np.ndarray
        Frequency array in Hz.
    mode_dict : dict
        Dictionary mapping (l, m) tuples to complex mode arrays h_lm(f).

    Examples
    --------
    >>> params = WaveformParameters(mass_1=30.0, mass_2=25.0, chi1z=0.3)
    >>> freqs, modes = generate_fd_modes(params, modes=[(2, 2), (3, 3)])
    >>> h22 = modes[(2, 2)]
    >>> h33 = modes[(3, 3)]
    """
    if modes is None:
        modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]

    mode_dict = {}
    frequencies = None

    for ell, emm in modes:
        freqs, h_lm = generate_fd_mode(params, ell, emm)
        mode_dict[(ell, emm)] = h_lm
        if frequencies is None:
            frequencies = freqs

    return frequencies, mode_dict
