"""Quantum states for the harmonic oscillator."""

import numpy as np
from typing import Union, Optional
from scipy.special import factorial


class QuantumState:
    """Base class for quantum states."""
    
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
    """Fock state |n⟩ (number eigenstate)."""
    
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
    """Coherent state |α⟩."""
    
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
    """Squeezed coherent state."""
    
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
    """Thermal state (mixed state) at temperature T."""
    
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
        """Expected photon number ⟨n⟩."""
        return self.n_bar