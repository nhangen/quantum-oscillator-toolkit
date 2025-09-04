"""Decoherence models for quantum harmonic oscillator."""

import numpy as np
from scipy.linalg import expm
from typing import Optional, Callable
from .oscillator import HarmonicOscillator
from .states import QuantumState


class DecoherenceModel:
    """Base class for decoherence models."""
    
    def __init__(self, oscillator: HarmonicOscillator):
        """Initialize decoherence model.
        
        Args:
            oscillator: HarmonicOscillator instance
        """
        self.oscillator = oscillator
    
    def lindblad_operator(self, rho: np.ndarray, operators: list, 
                         rates: list, hamiltonian: np.ndarray, dt: float) -> np.ndarray:
        """General Lindblad master equation evolution.
        
        Args:
            rho: Density matrix
            operators: List of Lindblad operators
            rates: List of decay rates
            hamiltonian: System Hamiltonian
            dt: Time step
            
        Returns:
            Updated density matrix
        """
        # Unitary evolution: -i/ℏ [H, ρ]
        commutator = hamiltonian @ rho - rho @ hamiltonian
        unitary_term = -1j * commutator / self.oscillator.hbar
        
        # Dissipative evolution: Σ γ_k (L_k ρ L_k† - 1/2 {L_k† L_k, ρ})
        dissipative_term = np.zeros_like(rho)
        
        for L_k, gamma_k in zip(operators, rates):
            L_k_dag = np.conj(L_k.T)
            
            # L_k ρ L_k†
            jump_term = L_k @ rho @ L_k_dag
            
            # 1/2 {L_k† L_k, ρ}
            anticommutator = 0.5 * (L_k_dag @ L_k @ rho + rho @ L_k_dag @ L_k)
            
            dissipative_term += gamma_k * (jump_term - anticommutator)
        
        return rho + dt * (unitary_term + dissipative_term)


class AmplitudeDamping(DecoherenceModel):
    """Amplitude damping (energy dissipation) model."""
    
    def __init__(self, oscillator: HarmonicOscillator, damping_rate: float):
        """Initialize amplitude damping.
        
        Args:
            oscillator: HarmonicOscillator instance  
            damping_rate: Energy damping rate γ
        """
        super().__init__(oscillator)
        self.damping_rate = damping_rate
    
    def evolve_density_matrix(self, rho: np.ndarray, time: float, 
                            n_steps: int = 100) -> np.ndarray:
        """Evolve density matrix with amplitude damping.
        
        Args:
            rho: Initial density matrix
            time: Total evolution time
            n_steps: Number of time steps
            
        Returns:
            Final density matrix
        """
        dt = time / n_steps
        n_max = rho.shape[0] - 1
        
        # Lindblad operator: L = √γ * a (annihilation operator)
        a = self.oscillator.annihilation_operator(n_max)
        H = self.oscillator.hamiltonian(n_max)
        
        current_rho = rho.copy()
        
        for _ in range(n_steps):
            current_rho = self.lindblad_operator(
                current_rho, [a], [self.damping_rate], H, dt
            )
        
        return current_rho


class PhaseDamping(DecoherenceModel):
    """Phase damping (pure dephasing) model."""
    
    def __init__(self, oscillator: HarmonicOscillator, dephasing_rate: float):
        """Initialize phase damping.
        
        Args:
            oscillator: HarmonicOscillator instance
            dephasing_rate: Dephasing rate γ_φ  
        """
        super().__init__(oscillator)
        self.dephasing_rate = dephasing_rate
    
    def evolve_density_matrix(self, rho: np.ndarray, time: float,
                            n_steps: int = 100) -> np.ndarray:
        """Evolve density matrix with phase damping.
        
        Args:
            rho: Initial density matrix
            time: Total evolution time  
            n_steps: Number of time steps
            
        Returns:
            Final density matrix
        """
        dt = time / n_steps
        n_max = rho.shape[0] - 1
        
        # Lindblad operator: L = √γ_φ * (a†a) (number operator)
        n_op = self.oscillator.number_operator(n_max)
        H = self.oscillator.hamiltonian(n_max)
        
        current_rho = rho.copy()
        
        for _ in range(n_steps):
            current_rho = self.lindblad_operator(
                current_rho, [n_op], [self.dephasing_rate], H, dt
            )
        
        return current_rho


class ThermalReservoir(DecoherenceModel):
    """Thermal reservoir coupling model."""
    
    def __init__(self, oscillator: HarmonicOscillator, coupling_strength: float, 
                 temperature: float, hbar: float = 1.0, kb: float = 1.0):
        """Initialize thermal reservoir coupling.
        
        Args:
            oscillator: HarmonicOscillator instance
            coupling_strength: System-bath coupling strength
            temperature: Bath temperature
            hbar: Reduced Planck constant
            kb: Boltzmann constant
        """
        super().__init__(oscillator)
        self.coupling_strength = coupling_strength
        self.temperature = temperature
        self.hbar = hbar
        self.kb = kb
        
        # Thermal occupation number
        if temperature > 0:
            self.n_th = 1.0 / (np.exp(hbar * oscillator.frequency / (kb * temperature)) - 1)
        else:
            self.n_th = 0.0
    
    def evolve_density_matrix(self, rho: np.ndarray, time: float,
                            n_steps: int = 100) -> np.ndarray:
        """Evolve density matrix with thermal reservoir.
        
        Args:
            rho: Initial density matrix
            time: Total evolution time
            n_steps: Number of time steps
            
        Returns:
            Final density matrix  
        """
        dt = time / n_steps
        n_max = rho.shape[0] - 1
        
        # Lindblad operators and rates for thermal bath
        a = self.oscillator.annihilation_operator(n_max)
        a_dag = self.oscillator.creation_operator(n_max)
        H = self.oscillator.hamiltonian(n_max)
        
        # Downward transitions: rate = γ(n_th + 1)  
        # Upward transitions: rate = γ*n_th
        gamma = self.coupling_strength
        
        operators = [a, a_dag]
        rates = [gamma * (self.n_th + 1), gamma * self.n_th]
        
        current_rho = rho.copy()
        
        for _ in range(n_steps):
            current_rho = self.lindblad_operator(current_rho, operators, rates, H, dt)
        
        return current_rho


class CompositeDecoherence(DecoherenceModel):
    """Composite decoherence model combining multiple effects."""
    
    def __init__(self, oscillator: HarmonicOscillator, 
                 amplitude_damping_rate: float = 0.0,
                 dephasing_rate: float = 0.0,
                 thermal_coupling: Optional[float] = None,
                 temperature: float = 0.0):
        """Initialize composite decoherence model.
        
        Args:
            oscillator: HarmonicOscillator instance
            amplitude_damping_rate: Energy damping rate
            dephasing_rate: Pure dephasing rate  
            thermal_coupling: Thermal reservoir coupling (None to disable)
            temperature: Bath temperature
        """
        super().__init__(oscillator)
        
        # Component models
        self.models = []
        
        if amplitude_damping_rate > 0:
            self.models.append(AmplitudeDamping(oscillator, amplitude_damping_rate))
        
        if dephasing_rate > 0:
            self.models.append(PhaseDamping(oscillator, dephasing_rate))
        
        if thermal_coupling is not None and thermal_coupling > 0:
            self.models.append(ThermalReservoir(oscillator, thermal_coupling, temperature))
    
    def evolve_density_matrix(self, rho: np.ndarray, time: float,
                            n_steps: int = 100) -> np.ndarray:
        """Evolve with combined decoherence effects.
        
        Uses Trotter decomposition: exp((A+B)t) ≈ [exp(At/n)exp(Bt/n)]^n
        
        Args:
            rho: Initial density matrix
            time: Total evolution time
            n_steps: Number of Trotter steps
            
        Returns:
            Final density matrix
        """
        if not self.models:
            return rho
        
        dt = time / n_steps
        current_rho = rho.copy()
        
        for _ in range(n_steps):
            # Apply each decoherence model sequentially (Trotter decomposition)
            for model in self.models:
                current_rho = model.evolve_density_matrix(current_rho, dt, n_steps=1)
        
        return current_rho


def coherence_measures(rho: np.ndarray) -> dict:
    """Calculate various coherence measures for a density matrix.
    
    Args:
        rho: Density matrix
        
    Returns:
        Dictionary of coherence measures
    """
    # Purity: Tr(ρ²)
    purity = np.real(np.trace(rho @ rho))
    
    # Linear entropy: S_L = 1 - Tr(ρ²)  
    linear_entropy = 1.0 - purity
    
    # Von Neumann entropy: S = -Tr(ρ log ρ)
    eigenvals = np.real(np.linalg.eigvals(rho))
    eigenvals = eigenvals[eigenvals > 1e-12]  # Remove zeros
    von_neumann_entropy = -np.sum(eigenvals * np.log(eigenvals))
    
    # Off-diagonal coherence (sum of squared off-diagonal elements)
    n_dim = rho.shape[0]
    off_diagonal_coherence = 0.0
    for i in range(n_dim):
        for j in range(n_dim):
            if i != j:
                off_diagonal_coherence += np.abs(rho[i, j])**2
    
    return {
        'purity': purity,
        'linear_entropy': linear_entropy, 
        'von_neumann_entropy': von_neumann_entropy,
        'off_diagonal_coherence': off_diagonal_coherence
    }