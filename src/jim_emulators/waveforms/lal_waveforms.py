"""
LAL waveform generation wrapper.

Provides a clean, testable interface to LALSimulation frequency-domain waveforms.
All physical quantities use SI units internally, with convenient input in
astrophysical units (solar masses, Mpc).
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a frequency-domain waveform using LALSimulation.

    Parameters
    ----------
    params : WaveformParameters
        Waveform parameters dataclass.

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
    """
    # Convert to SI units
    mass_1_si = params.mass_1 * MSUN_SI
    mass_2_si = params.mass_2 * MSUN_SI
    distance_si = params.luminosity_distance * MPC_SI

    # Get approximant enum
    approximant = SUPPORTED_APPROXIMANTS[params.approximant]

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
        None,  # LAL dictionary for extra parameters
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
