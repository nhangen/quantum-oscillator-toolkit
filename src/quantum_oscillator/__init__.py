"""Quantum Oscillator Toolkit

A comprehensive Python toolkit for quantum harmonic oscillator simulation,
coherent state manipulation, and decoherence modeling. This package provides
numerically efficient implementations of fundamental quantum mechanics concepts
essential for quantum optics, condensed matter physics, and quantum information.

The toolkit includes:
    - Exact harmonic oscillator operators and energy spectra
    - Coherent, Fock, squeezed, and thermal quantum states
    - Unitary and stochastic time evolution
    - Lindblad master equation decoherence models
    - Classical-quantum correspondence analysis
    - Quantum coherence and entanglement measures

Designed for researchers and students studying quantum systems, this toolkit
combines theoretical rigor with computational efficiency to enable exploration
of quantum phenomena across energy scales and decoherence regimes.

Author: @nhangen
Version: 0.1.0
License: Academic Research Use
"""

from .oscillator import HarmonicOscillator
from .states import CoherentState, FockState
from .evolution import TimeEvolution
from .decoherence import DecoherenceModel

__version__ = "0.1.0"
__all__ = ["HarmonicOscillator", "CoherentState", "FockState", "TimeEvolution", "DecoherenceModel"]