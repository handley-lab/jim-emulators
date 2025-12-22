"""
Waveform generation and comparison utilities.

This module provides:
- LAL waveform wrapper for generating reference waveforms
- Match/mismatch calculations
- Frequency grid utilities
- Time domain conversion (physical and geometric)
"""

from .lal_waveforms import (
    generate_fd_waveform,
    generate_fd_mode,
    generate_fd_modes,
    WaveformParameters,
    SUPPORTED_APPROXIMANTS,
    # Mode selection
    create_mode_array,
    get_available_modes,
    DEFAULT_MODES_XPHM,
    DEFAULT_MODES_XHM,
    DEFAULT_MODES_XAS,
)
from .utils import (
    get_frequency_array,
    noise_weighted_inner_product,
    compute_match,
    compute_mismatch,
    # Time domain
    fd_to_td,
    fd_to_td_centered_at_merger,
    geometric_fd_to_td,
    geometric_time_to_physical,
    physical_time_to_geometric,
    interpolate_to_uniform_grid,
    # Geometric frequency
    get_geometric_frequency_grid,
    geometric_to_physical_frequency,
    physical_to_geometric_frequency,
)
