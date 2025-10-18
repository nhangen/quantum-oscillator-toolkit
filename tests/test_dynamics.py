"""Tests for quantum harmonic oscillator dynamics and time evolution."""

import pytest
import numpy as np
from quantum_oscillator import HarmonicOscillator, CoherentState, FockState
from quantum_oscillator.evolution import TimeEvolution


class TestOscillatorDynamics:
    """Test cases for harmonic oscillator time evolution."""
    
    def test_coherent_state_oscillation(self):
        # Test that coherent states exhibit sinusoidal oscillation in phase space
        oscillator = HarmonicOscillator(frequency=1.0, mass=1.0, hbar=1.0)
        evolution = TimeEvolution(oscillator)
        
        # Create coherent state with real displacement (along position axis)
        alpha = 2.0  # Real coherent state parameter
        coherent_state = CoherentState(alpha, n_max=20)
        
        # Time points for one complete oscillation (T = 2π/ω = 2π for ω=1)
        times = np.linspace(0, 2*np.pi, 100)
        
        positions = []
        momenta = []
        
        for t in times:
            pos = evolution.expectation_position(coherent_state, t)
            mom = evolution.expectation_momentum(coherent_state, t)
            positions.append(pos)
            momenta.append(mom)
        
        positions = np.array(positions)
        momenta = np.array(momenta)
        
        # Check sinusoidal oscillation: x(t) = A cos(ωt + φ)
        # For α = 2.0 (real), initial position = √2 * 2 = 2√2, initial momentum = 0
        expected_amplitude = np.sqrt(2) * abs(alpha)
        
        # Position should oscillate as cos(t) (starting at maximum)
        expected_positions = expected_amplitude * np.cos(times)
        
        # Momentum should oscillate as -sin(t) (starting at zero)
        expected_momenta = -expected_amplitude * np.sin(times)
        
        # Check oscillation amplitudes and phases
        assert np.isclose(np.max(positions), expected_amplitude, rtol=0.01)
        assert np.isclose(np.min(positions), -expected_amplitude, rtol=0.01)
        assert np.isclose(positions[0], expected_amplitude, rtol=0.01)  # Initial position
        assert np.isclose(momenta[0], 0.0, atol=0.01)  # Initial momentum
        
        # Check period: position should return to initial value after T = 2π
        assert np.isclose(positions[-1], positions[0], rtol=0.01)
        assert np.isclose(momenta[-1], momenta[0], rtol=0.01)
        
        # Check sinusoidal form (correlation with expected)
        pos_correlation = np.corrcoef(positions, expected_positions)[0, 1]
        mom_correlation = np.corrcoef(momenta, expected_momenta)[0, 1]
        
        assert pos_correlation > 0.99  # Nearly perfect sinusoidal
        assert mom_correlation > 0.99
    
    def test_classical_quantum_correspondence(self):
        # Test that quantum oscillator matches classical trajectory for coherent states
        oscillator = HarmonicOscillator(frequency=2.0, mass=1.0, hbar=1.0)
        evolution = TimeEvolution(oscillator)
        
        # Complex coherent state parameter
        alpha = 1.0 + 0.5j
        coherent_state = CoherentState(alpha, n_max=15)
        
        # Initial classical conditions from coherent state
        # For our units: ⟨x⟩ = √2 * Re(α) * length_scale, ⟨p⟩ = √2 * Im(α) * momentum_scale  
        length_scale = oscillator.length_scale
        momentum_scale = np.sqrt(oscillator.mass * oscillator.hbar * oscillator.frequency)
        
        x0 = np.sqrt(2) * np.real(alpha) * length_scale
        p0 = np.sqrt(2) * np.imag(alpha) * momentum_scale
        
        times = np.linspace(0, np.pi, 50)  # Half period for ω=2
        
        # Quantum expectation values
        quantum_positions = []
        quantum_momenta = []
        
        for t in times:
            pos = evolution.expectation_position(coherent_state, t)
            mom = evolution.expectation_momentum(coherent_state, t)
            quantum_positions.append(pos)
            quantum_momenta.append(mom)
        
        # Classical trajectory
        classical_positions, classical_momenta = evolution.classical_trajectory(x0, p0, times)
        
        # Quantum and classical should match exactly for coherent states
        np.testing.assert_allclose(quantum_positions, classical_positions, rtol=0.01)
        np.testing.assert_allclose(quantum_momenta, classical_momenta, rtol=0.01)
    
    def test_fock_state_stationary(self):
        # Test that Fock states are stationary (no oscillation)
        oscillator = HarmonicOscillator(frequency=1.0)
        evolution = TimeEvolution(oscillator)
        
        # Ground state |0⟩
        ground_state = FockState(0, n_max=10)
        
        times = np.linspace(0, 4*np.pi, 50)  # Multiple periods
        
        positions = []
        momenta = []
        
        for t in times:
            pos = evolution.expectation_position(ground_state, t)
            mom = evolution.expectation_momentum(ground_state, t)
            positions.append(pos)
            momenta.append(mom)
        
        # Fock states should have zero expectation values (centered at origin)
        np.testing.assert_allclose(positions, 0.0, atol=1e-10)
        np.testing.assert_allclose(momenta, 0.0, atol=1e-10)
    
    def test_energy_conservation(self):
        # Test that energy is conserved during unitary evolution
        oscillator = HarmonicOscillator(frequency=1.5)
        evolution = TimeEvolution(oscillator)
        
        # Mixed initial state
        alpha = 1.5 - 0.8j
        initial_state = CoherentState(alpha, n_max=12)
        
        # Initial energy
        H = oscillator.hamiltonian(12)
        initial_rho = initial_state.density_matrix()
        initial_energy = np.real(np.trace(H @ initial_rho))
        
        # Evolve and check energy at different times
        times = [0.5, 1.0, 2.0, 5.0]
        
        for t in times:
            evolved_state = evolution.evolve_state(initial_state, t)
            evolved_rho = evolved_state.density_matrix()
            evolved_energy = np.real(np.trace(H @ evolved_rho))
            
            # Energy should be conserved
            assert np.isclose(evolved_energy, initial_energy, rtol=1e-6)
    
    def test_phase_space_circular_motion(self):
        # Test that coherent states trace circular trajectories in phase space
        oscillator = HarmonicOscillator(frequency=1.0)
        evolution = TimeEvolution(oscillator)
        
        # Circular motion requires complex α
        alpha = 2.0 * np.exp(1j * np.pi/4)  # 45-degree initial phase
        coherent_state = CoherentState(alpha, n_max=20)
        
        times = np.linspace(0, 2*np.pi, 100)  # One complete orbit
        
        positions = []
        momenta = []
        
        for t in times:
            pos = evolution.expectation_position(coherent_state, t)
            mom = evolution.expectation_momentum(coherent_state, t)
            positions.append(pos)
            momenta.append(mom)
        
        positions = np.array(positions)
        momenta = np.array(momenta)
        
        # Calculate distance from origin (should be constant for circular motion)
        radius = np.sqrt(positions**2 + momenta**2)
        expected_radius = np.sqrt(2) * abs(alpha)  # √2 |α|
        
        # All points should be at the same radius
        np.testing.assert_allclose(radius, expected_radius, rtol=0.01)
        
        # Should complete exactly one circle (return to start)
        assert np.isclose(positions[0], positions[-1], rtol=0.01)
        assert np.isclose(momenta[0], momenta[-1], rtol=0.01)
    
    def test_frequency_scaling(self):
        # Test that oscillation frequency scales correctly
        frequencies = [0.5, 1.0, 2.0, 4.0]
        alpha = 1.0  # Real coherent state
        
        for omega in frequencies:
            oscillator = HarmonicOscillator(frequency=omega)
            evolution = TimeEvolution(oscillator)
            coherent_state = CoherentState(alpha, n_max=15)
            
            # Period should be T = 2π/ω
            expected_period = 2*np.pi / omega
            
            # Sample over one expected period
            times = np.linspace(0, expected_period, 100)
            positions = []
            
            for t in times:
                pos = evolution.expectation_position(coherent_state, t)
                positions.append(pos)
            
            # Position should return to initial value after one period
            assert np.isclose(positions[0], positions[-1], rtol=0.01)
            
            # Check that it's actually oscillating (not constant)
            position_range = np.max(positions) - np.min(positions)
            assert position_range > 0.1  # Significant oscillation
    
    def test_coherent_state_parameter_evolution(self):
        # Test that coherent state parameter evolves as α(t) = α(0) exp(-iωt)
        oscillator = HarmonicOscillator(frequency=2.0)
        evolution = TimeEvolution(oscillator)
        
        alpha_0 = 1.0 + 2.0j
        times = np.linspace(0, np.pi, 25)  # Quarter period for ω=2
        
        # Analytical evolution
        expected_alphas = evolution.coherent_state_trajectory(alpha_0, times)
        
        # Check evolution law: α(t) = α(0) exp(-iωt)
        omega = oscillator.frequency
        analytical_alphas = alpha_0 * np.exp(-1j * omega * times)
        
        np.testing.assert_allclose(expected_alphas, analytical_alphas, rtol=1e-10)
        
        # Magnitude should be conserved
        magnitudes = np.abs(expected_alphas)
        expected_magnitude = abs(alpha_0)
        
        np.testing.assert_allclose(magnitudes, expected_magnitude, rtol=1e-10)