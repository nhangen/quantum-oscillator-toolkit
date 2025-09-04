"""Quantum Harmonic Oscillator implementation."""

import numpy as np
from scipy import constants
from typing import Optional


class HarmonicOscillator:
    """Quantum harmonic oscillator with creation/annihilation operators."""
    
    def __init__(self, frequency: float = 1.0, mass: float = 1.0, hbar: float = 1.0):
        """Initialize quantum harmonic oscillator.
        
        Args:
            frequency: Oscillator frequency ω
            mass: Oscillator mass m  
            hbar: Reduced Planck constant (default: 1.0 for natural units)
        """
        self.frequency = frequency
        self.mass = mass
        self.hbar = hbar
        
        # Characteristic length scale
        self.length_scale = np.sqrt(self.hbar / (self.mass * self.frequency))
        
        # Energy scale
        self.energy_scale = self.hbar * self.frequency
        
    def energy_eigenvalue(self, n: int) -> float:
        """Energy eigenvalue for Fock state |n⟩.
        
        Args:
            n: Fock state number (non-negative integer)
            
        Returns:
            Energy eigenvalue E_n = ℏω(n + 1/2)
        """
        if n < 0:
            raise ValueError("Fock state number must be non-negative")
        return self.energy_scale * (n + 0.5)
    
    def creation_operator(self, n_max: int) -> np.ndarray:
        """Matrix representation of creation operator a†.
        
        Args:
            n_max: Maximum Fock state to include
            
        Returns:
            Matrix representation of a† in Fock basis
        """
        dim = n_max + 1
        a_dag = np.zeros((dim, dim), dtype=complex)
        
        for n in range(n_max):
            a_dag[n + 1, n] = np.sqrt(n + 1)
            
        return a_dag
    
    def annihilation_operator(self, n_max: int) -> np.ndarray:
        """Matrix representation of annihilation operator a.
        
        Args:
            n_max: Maximum Fock state to include
            
        Returns:
            Matrix representation of a in Fock basis
        """
        dim = n_max + 1
        a = np.zeros((dim, dim), dtype=complex)
        
        for n in range(1, dim):
            a[n - 1, n] = np.sqrt(n)
            
        return a
    
    def number_operator(self, n_max: int) -> np.ndarray:
        """Matrix representation of number operator n = a†a.
        
        Args:
            n_max: Maximum Fock state to include
            
        Returns:
            Matrix representation of number operator
        """
        return np.diag(range(n_max + 1))
    
    def hamiltonian(self, n_max: int) -> np.ndarray:
        """Matrix representation of Hamiltonian H = ℏω(a†a + 1/2).
        
        Args:
            n_max: Maximum Fock state to include
            
        Returns:
            Hamiltonian matrix in Fock basis
        """
        n_op = self.number_operator(n_max)
        return self.energy_scale * (n_op + 0.5 * np.eye(n_max + 1))
    
    def position_operator(self, n_max: int) -> np.ndarray:
        """Matrix representation of position operator x.
        
        Args:
            n_max: Maximum Fock state to include
            
        Returns:
            Position operator matrix
        """
        a = self.annihilation_operator(n_max)
        a_dag = self.creation_operator(n_max)
        
        return self.length_scale / np.sqrt(2) * (a + a_dag)
    
    def momentum_operator(self, n_max: int) -> np.ndarray:
        """Matrix representation of momentum operator p.
        
        Args:
            n_max: Maximum Fock state to include
            
        Returns:
            Momentum operator matrix
        """
        a = self.annihilation_operator(n_max)
        a_dag = self.creation_operator(n_max)
        
        return 1j * np.sqrt(self.mass * self.hbar * self.frequency / 2) * (a_dag - a)