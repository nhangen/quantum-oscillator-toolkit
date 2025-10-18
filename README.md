# Quantum Oscillator Toolkit

A comprehensive Python toolkit for quantum harmonic oscillator simulation, coherent state manipulation, and decoherence modeling. This package provides numerically efficient implementations of fundamental quantum mechanics concepts essential for quantum optics, condensed matter physics, and quantum information.

## Features

### Core Quantum Mechanics
- **Harmonic Oscillator Operators**: Exact matrix representations of creation/annihilation operators, position/momentum operators, and the Hamiltonian in the Fock basis
- **Energy Eigenvalues & Eigenstates**: Precise calculation of energy levels E_n = ℏω(n + 1/2) and Fock state representations
- **Canonical Commutation Relations**: Verified implementation of [x̂, p̂] = iℏ and [â, â†] = 1

### Quantum States
- **Fock States**: Energy eigenstates |n⟩ with definite excitation number
- **Coherent States**: Minimum uncertainty states |α⟩ exhibiting classical-like behavior
- **Squeezed States**: Reduced uncertainty in one quadrature for enhanced measurement precision
- **Thermal States**: Mixed states representing thermal equilibrium at finite temperature

### Time Evolution & Dynamics
- **Unitary Evolution**: Exact time evolution via U(t) = exp(-iĤt/ℏ)
- **Classical Trajectories**: Comparison with classical harmonic motion
- **Coherent State Evolution**: α(t) = α₀ exp(-iωt) parameter evolution
- **Stochastic Evolution**: Environmental noise effects on quantum dynamics

### Decoherence Modeling
- **Lindblad Master Equation**: Implementation based on proven DeCoN PINN solvers
- **Amplitude Damping**: Energy dissipation and spontaneous emission
- **Phase Damping**: Pure dephasing without energy loss
- **Thermal Reservoirs**: Finite temperature environmental coupling
- **Coherence Measures**: Purity, von Neumann entropy, and off-diagonal coherence

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from quantum_oscillator import HarmonicOscillator, CoherentState, FockState
from quantum_oscillator.evolution import TimeEvolution
from quantum_oscillator.decoherence import AmplitudeDamping

# Create quantum harmonic oscillator (natural units: ℏ = m = ω = 1)
oscillator = HarmonicOscillator(frequency=1.0, mass=1.0, hbar=1.0)

# Generate coherent state |α⟩ with complex amplitude α
alpha = 2.0 + 1.0j
coherent_state = CoherentState(alpha, n_max=20)

# Time evolution
evolution = TimeEvolution(oscillator)
time = np.pi  # Half period for ω = 1
evolved_state = evolution.evolve_state(coherent_state, time)

# Calculate expectation values
x_expect = evolution.expectation_position(coherent_state, time)
p_expect = evolution.expectation_momentum(coherent_state, time)

# Model decoherence with amplitude damping
damping = AmplitudeDamping(oscillator, damping_rate=0.1)
rho_initial = coherent_state.density_matrix()
rho_final = damping.evolve_density_matrix(rho_initial, time=5.0)
```

## Project Structure

```
quantum-oscillator-toolkit/
├── src/quantum_oscillator/
│   ├── __init__.py           # Package initialization
│   ├── oscillator.py         # Core harmonic oscillator operators
│   ├── states.py             # Quantum state representations
│   ├── evolution.py          # Time evolution operators
│   └── decoherence.py        # DeCoN PINN-based Lindblad solvers
├── examples/
│   ├── basic_usage.py        # Tutorial examples
│   └── plot_equations.py     # Visualization of quantum dynamics
├── tests/
│   ├── test_oscillator.py    # Operator algebra tests
│   ├── test_states.py        # State property verification
│   └── test_dynamics.py      # Evolution and decoherence tests
└── docs/
    └── EQUATIONS.md          # Mathematical reference

## Mathematical Foundation

This toolkit implements the quantum harmonic oscillator Hamiltonian:

```
Ĥ = ℏω(â†â + 1/2)
```

where `â†` and `â` are the creation and annihilation operators satisfying `[â, â†] = 1`.

Key equations include:
- Energy eigenvalues: `E_n = ℏω(n + 1/2)`
- Position operator: `x̂ = (ℓ/√2)(â + â†)` where `ℓ = √(ℏ/mω)`
- Momentum operator: `p̂ = i√(mℏω/2)(â† - â)`
- Coherent state evolution: `|α(t)⟩ = |α₀ e^(-iωt)⟩`

## Examples

See `examples/basic_usage.py` for comprehensive demonstrations including:
- Coherent state time evolution and phase space trajectories
- Fock state energy spectra and uncertainty relations
- Decoherence dynamics under various environmental models
- Quantum state visualization and comparison

## Testing

Run the test suite to verify implementation correctness:

```bash
pytest tests/
```

## Author

@nhangen

## License

Academic Research Use