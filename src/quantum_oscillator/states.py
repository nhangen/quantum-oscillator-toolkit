"""Quantum state representations for the harmonic oscillator.

This module provides implementations of fundamental quantum states that arise
in harmonic oscillator systems. Each state class encapsulates the mathematical
structure and physical properties of important quantum mechanical states.

Supported state types:
    - Fock states |n⟩: Energy eigenstates with definite excitation number
    - Coherent states |α⟩: Minimum uncertainty states exhibiting classical behavior
    - Squeezed states: Reduced uncertainty in one quadrature at the cost of increased
      uncertainty in the conjugate quadrature
    - Thermal states: Mixed states representing thermal equilibrium at finite temperature

These states form the foundation for quantum optics calculations, cavity QED
simulations, and studies of quantum-classical correspondence.

References:
    Quantum Optics, M.O. Scully & M.S. Zubairy
    Introductory Quantum Optics, C.C. Gerry & P.L. Knight
    Quantum Theory of Light, R. Loudon
"""

import numpy as np
from typing import Union, Optional
from scipy.special import factorial


class QuantumState:
    """Abstract base class for quantum state representations.
    
    Provides the common interface and fundamental operations for all quantum
    states in the harmonic oscillator Hilbert space. States are represented
    in the Fock basis {|0⟩, |1⟩, |2⟩, ...} with complex amplitudes.
    
    The state vector |ψ⟩ = Σₙ cₙ|n⟩ is stored as the amplitude array {cₙ},
    automatically normalized to ensure ⟨ψ|ψ⟩ = 1.
    
    All quantum states support:
        - Probability calculations for Fock state measurements
        - Expectation values of quantum observables
        - Density matrix construction for mixed state analysis
        - Normalization and dimension queries
    
    This abstract interface enables polymorphic treatment of different
    quantum state types in evolution and decoherence calculations.
    """
    
    def __init__(self, amplitudes: np.ndarray):
        """Initialize quantum state.
        
        Args:
            amplitudes: State vector in Fock basis
        """
        self.amplitudes = np.array(amplitudes, dtype=complex)
        self._normalize()
    
    def _normalize(self):
        """Normalize the state vector."""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
        if norm > 1e-12:
            self.amplitudes /= norm
    
    @property
    def dimension(self) -> int:
        """Hilbert space dimension (number of Fock states)."""
        return len(self.amplitudes)
    
    def probability(self, n: int) -> float:
        """Probability of measuring Fock state |n⟩.
        
        Args:
            n: Fock state number
            
        Returns:
            Probability |⟨n|ψ⟩|²
        """
        if n >= self.dimension:
            return 0.0
        return np.abs(self.amplitudes[n])**2
    
    def expectation_number(self) -> float:
        """Expectation value of number operator ⟨n⟩."""
        return sum(n * self.probability(n) for n in range(self.dimension))
    
    def density_matrix(self) -> np.ndarray:
        """Density matrix representation ρ = |ψ⟩⟨ψ|."""
        return np.outer(self.amplitudes, np.conj(self.amplitudes))


class FockState(QuantumState):
    """Fock state |n⟩ representing definite excitation number.
    
    Fock states are the energy eigenstates of the quantum harmonic oscillator
    with definite excitation number n. They form an orthonormal basis for the
    infinite-dimensional Hilbert space:
    
        ⟨m|n⟩ = δₘₙ
        Ĥ|n⟩ = ℏω(n + 1/2)|n⟩
    
    Key properties:
        - Well-defined energy: E_n = ℏω(n + 1/2)
        - Maximum uncertainty in position and momentum
        - Zero mean position and momentum: ⟨x⟩ = ⟨p⟩ = 0
        - Stationary under time evolution (only acquire phase factors)
    
    Fock states are fundamental in quantum optics where they represent
    definite photon number states, and in cavity QED for discrete energy
    level systems.
    
    Attributes:
        n (int): The excitation number (non-negative integer)
    """
    
    def __init__(self, n: int, n_max: int):
        """Initialize Fock state.
        
        Args:
            n: Fock state number
            n_max: Maximum Fock state in basis
        """
        if n < 0 or n > n_max:
            raise ValueError(f"Fock state number {n} out of range [0, {n_max}]")
        
        amplitudes = np.zeros(n_max + 1, dtype=complex)
        amplitudes[n] = 1.0
        super().__init__(amplitudes)
        
        self.n = n


class CoherentState(QuantumState):
    """Coherent state |α⟩ with minimum uncertainty and classical behavior.
    
    Coherent states are the closest quantum analogs to classical harmonic
    motion. They minimize the Heisenberg uncertainty relation and exhibit
    classical trajectories in phase space under time evolution.
    
    Mathematical definition:
        |α⟩ = e^(-|α|²/2) Σₙ (αⁿ/√n!) |n⟩
    
    Key properties:
        - Minimum uncertainty: Δx Δp = ℏ/2
        - Classical expectation values: ⟨x⟩ = √2 Re(α), ⟨p⟩ = √2 Im(α)
        - Poisson photon number distribution
        - Time evolution: |α(t)⟩ = |α₀ e^(-iωt)⟩
        - Circular trajectories in phase space
    
    Coherent states are eigenvalues of the annihilation operator:
        â|α⟩ = α|α⟩
    
    They play central roles in:
        - Laser physics (stable light field states)
        - Quantum optics (semiclassical limit)
        - Quantum communication (phase/amplitude encoding)
        - Quantum metrology (phase estimation protocols)
    
    Attributes:
        alpha (complex): The coherent state parameter α = |α|e^(iφ)
    """
    
    def __init__(self, alpha: complex, n_max: int = 20):
        """Initialize coherent state.
        
        Args:
            alpha: Complex coherent state parameter
            n_max: Maximum Fock state to include in expansion
        """
        self.alpha = complex(alpha)
        
        # Coherent state expansion: |α⟩ = exp(-|α|²/2) Σ αⁿ/√n! |n⟩
        amplitudes = np.zeros(n_max + 1, dtype=complex)
        normalization = np.exp(-0.5 * np.abs(alpha)**2)
        
        for n in range(n_max + 1):
            amplitudes[n] = normalization * (alpha**n) / np.sqrt(factorial(n))
        
        super().__init__(amplitudes)
    
    @property
    def amplitude(self) -> complex:
        """Coherent state amplitude α."""
        return self.alpha
    
    def displacement(self) -> float:
        """Classical displacement ⟨x⟩."""
        return np.sqrt(2) * np.real(self.alpha)  # In natural units
    
    def momentum_displacement(self) -> float:
        """Classical momentum ⟨p⟩."""
        return np.sqrt(2) * np.imag(self.alpha)  # In natural units


class SqueezedState(QuantumState):
    """Squeezed coherent state with reduced quadrature uncertainty.
    
    Squeezed states achieve uncertainty reduction in one quadrature (position
    or momentum) below the standard quantum limit, at the cost of increased
    uncertainty in the conjugate quadrature. They maintain the minimum
    uncertainty product Δx Δp = ℏ/2.
    
    The squeezing transformation is generated by:
        Ŝ(ξ) = exp[½(ξ*â² - ξ*â†²)]
    
    where ξ = re^(iθ) is the complex squeezing parameter.
    
    Physical applications:
        - Gravitational wave detection (reduced shot noise)
        - Quantum metrology (enhanced parameter estimation)
        - Quantum communication (increased channel capacity)
        - Atomic spectroscopy (reduced measurement uncertainty)
    
    Key properties:
        - Quadrature uncertainty: Δx₁ = e^(-r)/2, Δx₂ = e^r/2
        - Non-classical photon statistics
        - Enhanced sensitivity for phase measurements
        - Fragility to losses and decoherence
    
    Attributes:
        alpha (complex): Displacement parameter
        squeeze_param (complex): Squeezing parameter ξ = re^(iθ)
    
    Note:
        This implementation provides a simplified treatment focusing on
        squeezed vacuum states. Full arbitrary squeezed coherent states
        require more sophisticated mathematical machinery.
    """
    
    def __init__(self, alpha: complex, squeeze_param: complex, n_max: int = 20):
        """Initialize squeezed coherent state.
        
        Args:
            alpha: Displacement parameter
            squeeze_param: Squeezing parameter ξ = r*exp(iθ)
            n_max: Maximum Fock state to include
        """
        self.alpha = complex(alpha)
        self.squeeze_param = complex(squeeze_param)
        
        r = np.abs(squeeze_param)
        theta = np.angle(squeeze_param)
        
        # Squeezed state construction (simplified for even n)
        amplitudes = np.zeros(n_max + 1, dtype=complex)
        
        # This is a simplified implementation
        # Full squeezed state requires more complex calculation
        prefactor = np.exp(-0.5 * np.abs(alpha)**2) / np.sqrt(np.cosh(r))
        
        for n in range(0, n_max + 1, 2):  # Only even terms for |0⟩ squeezed vacuum
            if n == 0:
                amplitudes[n] = prefactor
            else:
                # Simplified squeezing coefficients
                amplitudes[n] = prefactor * np.sqrt(factorial(n)) * \
                              (-0.5 * np.exp(1j * theta) * np.tanh(r))**(n//2) / \
                              factorial(n//2)
        
        super().__init__(amplitudes)


class ThermalState:
    """Thermal equilibrium state at finite temperature.
    
    Represents the quantum harmonic oscillator in thermal equilibrium with
    a heat bath at temperature T. This is a mixed state (not a pure state)
    described by the canonical density matrix:
    
        ρ_th = Z⁻¹ exp(-Ĥ/kT)
    
    where Z = Tr[exp(-Ĥ/kT)] is the partition function.
    
    In the Fock basis, thermal states are diagonal with probabilities:
        P(n) = (n̄/(1+n̄))ⁿ × 1/(1+n̄)
    
    where n̄ = [exp(ℏω/kT) - 1]⁻¹ is the thermal occupation number.
    
    Physical properties:
        - Mean energy: ⟨Ĥ⟩ = ℏω(n̄ + 1/2)
        - Heat capacity: C = kB(ℏω/kT)² n̄(n̄+1)
        - Classical limit: n̄ → kT/ℏω for kT >> ℏω
        - Quantum limit: ground state for T → 0
    
    Thermal states are crucial for:
        - Statistical mechanics of quantum systems
        - Modeling realistic experimental conditions
        - Understanding thermodynamic properties
        - Decoherence studies with finite-temperature environments
    
    Attributes:
        temperature (float): System temperature T
        n_bar (float): Mean thermal occupation number n̄
        probabilities (np.ndarray): Fock state occupation probabilities
    """
    
    def __init__(self, temperature: float, frequency: float = 1.0, 
                 hbar: float = 1.0, kb: float = 1.0, n_max: int = 20):
        """Initialize thermal state.
        
        Args:
            temperature: Temperature
            frequency: Oscillator frequency
            hbar: Reduced Planck constant
            kb: Boltzmann constant
            n_max: Maximum Fock state
        """
        self.temperature = temperature
        self.frequency = frequency
        self.hbar = hbar
        self.kb = kb
        
        # Mean occupation number
        if temperature > 0:
            self.n_bar = 1.0 / (np.exp(hbar * frequency / (kb * temperature)) - 1)
        else:
            self.n_bar = 0.0
        
        # Thermal state probabilities
        self.probabilities = np.zeros(n_max + 1)
        
        if temperature > 0:
            for n in range(n_max + 1):
                self.probabilities[n] = (self.n_bar / (1 + self.n_bar))**n / (1 + self.n_bar)
        else:
            # Ground state at T=0
            self.probabilities[0] = 1.0
    
    def density_matrix(self) -> np.ndarray:
        """Thermal state density matrix."""
        dim = len(self.probabilities)
        rho = np.zeros((dim, dim), dtype=complex)
        
        for n in range(dim):
            rho[n, n] = self.probabilities[n]
        
        return rho
    
    def expectation_number(self) -> float:
        """Expected excitation number ⟨n̂⟩ for thermal state.
        
        Returns:
            float: Mean occupation number n̄ = [exp(ℏω/kT) - 1]⁻¹
        """
        return self.n_bar