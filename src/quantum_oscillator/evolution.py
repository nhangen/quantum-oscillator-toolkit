"""Time evolution for quantum harmonic oscillator."""

import numpy as np
from scipy.linalg import expm
from typing import Union
from .oscillator import HarmonicOscillator
from .states import QuantumState


class TimeEvolution:
    """Time evolution operator for quantum harmonic oscillator."""
    
    def __init__(self, oscillator: HarmonicOscillator):
        """Initialize time evolution.
        
        Args:
            oscillator: HarmonicOscillator instance
        """
        self.oscillator = oscillator
    
    def evolution_operator(self, time: float, n_max: int) -> np.ndarray:
        """Unitary time evolution operator U(t) = exp(-iHt/ℏ).
        
        Args:
            time: Evolution time
            n_max: Maximum Fock state dimension
            
        Returns:
            Unitary evolution operator matrix
        """
        hamiltonian = self.oscillator.hamiltonian(n_max)
        return expm(-1j * hamiltonian * time / self.oscillator.hbar)
    
    def evolve_state(self, initial_state: QuantumState, time: float) -> QuantumState:
        """Evolve quantum state in time.
        
        Args:
            initial_state: Initial quantum state
            time: Evolution time
            
        Returns:
            Evolved quantum state |ψ(t)⟩ = U(t)|ψ(0)⟩
        """
        n_max = initial_state.dimension - 1
        U_t = self.evolution_operator(time, n_max)
        
        evolved_amplitudes = U_t @ initial_state.amplitudes
        
        # Return new QuantumState with evolved amplitudes
        from .states import QuantumState
        return QuantumState(evolved_amplitudes)
    
    def classical_trajectory(self, initial_position: float, initial_momentum: float, 
                           times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Classical harmonic oscillator trajectory.
        
        Args:
            initial_position: Initial position x(0)
            initial_momentum: Initial momentum p(0) 
            times: Time points for trajectory
            
        Returns:
            Tuple of (positions, momenta) arrays
        """
        omega = self.oscillator.frequency
        m = self.oscillator.mass
        
        # Classical solution: x(t) = A*cos(ωt + φ), p(t) = -mωA*sin(ωt + φ)
        A = np.sqrt(initial_position**2 + (initial_momentum / (m * omega))**2)
        phi = np.arctan2(-initial_momentum / (m * omega), initial_position)
        
        positions = A * np.cos(omega * times + phi)
        momenta = -m * omega * A * np.sin(omega * times + phi)
        
        return positions, momenta
    
    def coherent_state_trajectory(self, alpha_0: complex, times: np.ndarray) -> np.ndarray:
        """Time evolution of coherent state parameter α(t).
        
        For coherent states, α(t) = α(0) * exp(-iωt)
        
        Args:
            alpha_0: Initial coherent state parameter
            times: Time points
            
        Returns:
            Array of α(t) values
        """
        omega = self.oscillator.frequency
        return alpha_0 * np.exp(-1j * omega * times)
    
    def expectation_position(self, state: QuantumState, time: float) -> float:
        """Time-dependent expectation value ⟨x(t)⟩.
        
        Args:
            state: Quantum state
            time: Time
            
        Returns:
            Position expectation value
        """
        evolved_state = self.evolve_state(state, time)
        n_max = evolved_state.dimension - 1
        
        x_op = self.oscillator.position_operator(n_max)
        rho = evolved_state.density_matrix()
        
        return np.real(np.trace(x_op @ rho))
    
    def expectation_momentum(self, state: QuantumState, time: float) -> float:
        """Time-dependent expectation value ⟨p(t)⟩.
        
        Args:
            state: Quantum state  
            time: Time
            
        Returns:
            Momentum expectation value
        """
        evolved_state = self.evolve_state(state, time)
        n_max = evolved_state.dimension - 1
        
        p_op = self.oscillator.momentum_operator(n_max)
        rho = evolved_state.density_matrix()
        
        return np.real(np.trace(p_op @ rho))


class StochasticEvolution(TimeEvolution):
    """Stochastic time evolution with noise."""
    
    def __init__(self, oscillator: HarmonicOscillator, noise_strength: float = 0.1):
        """Initialize stochastic evolution.
        
        Args:
            oscillator: HarmonicOscillator instance
            noise_strength: Strength of stochastic noise
        """
        super().__init__(oscillator)
        self.noise_strength = noise_strength
    
    def stochastic_evolution_operator(self, time: float, n_max: int, 
                                    random_seed: int = None) -> np.ndarray:
        """Stochastic evolution with random phase noise.
        
        Args:
            time: Evolution time
            n_max: Maximum Fock state
            random_seed: Random seed for reproducibility
            
        Returns:
            Stochastic evolution operator
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Base Hamiltonian evolution
        U_0 = self.evolution_operator(time, n_max)
        
        # Add random phase noise
        random_phases = self.noise_strength * np.random.normal(0, 1, n_max + 1)
        noise_operator = np.diag(np.exp(1j * random_phases))
        
        return noise_operator @ U_0