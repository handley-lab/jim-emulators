"""
Emulator components for gravitational waveform emulation.

This package provides:
- WaveformPCA: PCA compression for waveforms
- SpeculatorActivation: Smooth activation function for HMC compatibility
- EmulatorMLP: Neural network architecture
- GWEmulator: Complete emulator pipeline
"""

from .pca import WaveformPCA
from .network import SpeculatorActivation, EmulatorMLP
from .emulator import GWEmulator

__all__ = [
    "WaveformPCA",
    "SpeculatorActivation",
    "EmulatorMLP",
    "GWEmulator",
]
