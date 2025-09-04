"""Quantum Oscillator Toolkit

A Python toolkit for quantum harmonic oscillator simulation and coherent state manipulation.
"""

from .oscillator import HarmonicOscillator
from .states import CoherentState, FockState
from .evolution import TimeEvolution
from .decoherence import DecoherenceModel

__version__ = "0.1.0"
__all__ = ["HarmonicOscillator", "CoherentState", "FockState", "TimeEvolution", "DecoherenceModel"]