"""
Emulator components for gravitational waveform emulation.

This package provides:
- WaveformPCA: PCA compression for waveforms
- SpeculatorActivation: Smooth activation function for HMC compatibility
- EmulatorMLP: Neural network architecture
- GWEmulator: Complete emulator pipeline
- PerFrequencyNormalizer: Per-frequency output normalization
- CoefficientStandardizer: PCA coefficient standardization
- InputNormalizer: Input parameter normalization
"""

from .pca import WaveformPCA
from .network import SpeculatorActivation, EmulatorMLP
from .emulator import GWEmulator, create_train_state, train_step, eval_loss
from .normalization import (
    PerFrequencyNormalizer,
    CoefficientStandardizer,
    InputNormalizer,
)

__all__ = [
    "WaveformPCA",
    "SpeculatorActivation",
    "EmulatorMLP",
    "GWEmulator",
    "PerFrequencyNormalizer",
    "CoefficientStandardizer",
    "InputNormalizer",
    "create_train_state",
    "train_step",
    "eval_loss",
]
