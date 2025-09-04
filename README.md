# Quantum Oscillator Toolkit

A Python toolkit for quantum harmonic oscillator simulation and coherent state manipulation.

## Features

- Quantum harmonic oscillator dynamics
- Coherent state operations
- Decoherence modeling
- Educational examples and tutorials

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from quantum_oscillator import HarmonicOscillator, CoherentState

# Create oscillator
oscillator = HarmonicOscillator(frequency=1.0, mass=1.0)

# Generate coherent state
alpha = 2.0 + 1.0j
coherent_state = CoherentState(alpha)

# Time evolution
evolved_state = oscillator.evolve(coherent_state, time=1.0)
```

## Structure

- `src/quantum_oscillator/` - Core oscillator physics
- `examples/` - Usage examples and tutorials
- `tests/` - Test suite
- `docs/` - Documentation

## License

MIT License