"""Tests for quantum states."""

import pytest
import numpy as np
from quantum_oscillator.states import FockState, CoherentState, ThermalState


class TestFockState:
    """Test cases for FockState class."""
    
    def test_fock_state_creation(self):
        """Test Fock state initialization."""
        n = 2
        n_max = 5
        fock_state = FockState(n, n_max)
        
        assert fock_state.n == n
        assert fock_state.dimension == n_max + 1
        
        # Should have amplitude 1 at position n, 0 elsewhere
        expected_amplitudes = np.zeros(n_max + 1, dtype=complex)
        expected_amplitudes[n] = 1.0
        
        assert np.allclose(fock_state.amplitudes, expected_amplitudes)
    
    def test_fock_state_probabilities(self):
        """Test Fock state probability calculations."""
        n = 3
        n_max = 10
        fock_state = FockState(n, n_max)
        
        # Probability should be 1 for state |n⟩, 0 for others
        assert fock_state.probability(n) == 1.0
        for i in range(n_max + 1):
            if i != n:
                assert fock_state.probability(i) == 0.0
        
        # Expectation value of number operator should be n
        assert fock_state.expectation_number() == n
    
    def test_fock_state_errors(self):
        """Test Fock state error handling."""
        # Negative n
        with pytest.raises(ValueError):
            FockState(-1, 5)
        
        # n > n_max
        with pytest.raises(ValueError):
            FockState(10, 5)


class TestCoherentState:
    """Test cases for CoherentState class."""
    
    def test_coherent_state_creation(self):
        """Test coherent state initialization."""
        alpha = 2.0 + 1.0j
        n_max = 10
        
        coherent_state = CoherentState(alpha, n_max)
        
        assert coherent_state.amplitude == alpha
        assert coherent_state.dimension == n_max + 1
        
        # State should be normalized
        norm_squared = np.sum(np.abs(coherent_state.amplitudes)**2)
        assert np.isclose(norm_squared, 1.0)
    
    def test_coherent_state_poisson_distribution(self):
        """Test that coherent state has Poissonian photon statistics."""
        alpha = 2.0  # Real alpha for simplicity
        n_max = 20
        
        coherent_state = CoherentState(alpha, n_max)
        
        # Mean photon number should be |α|²
        mean_n = coherent_state.expectation_number()
        expected_mean = abs(alpha)**2
        assert np.isclose(mean_n, expected_mean, rtol=0.01)
        
        # Check Poissonian distribution (approximately for small n)
        for n in range(5):
            prob = coherent_state.probability(n)
            
            # Poissonian: P(n) = |α|²ⁿ exp(-|α|²) / n!
            from scipy.special import factorial
            expected_prob = (abs(alpha)**(2*n) * np.exp(-abs(alpha)**2) / factorial(n))
            
            assert np.isclose(prob, expected_prob, rtol=0.05)
    
    def test_coherent_state_displacement(self):
        """Test coherent state classical displacements."""
        alpha = 3.0 + 2.0j
        coherent_state = CoherentState(alpha, n_max=15)
        
        # Position displacement ⟨x⟩ = √2 Re(α) (natural units)
        x_displacement = coherent_state.displacement()
        expected_x = np.sqrt(2) * np.real(alpha)
        assert np.isclose(x_displacement, expected_x)
        
        # Momentum displacement ⟨p⟩ = √2 Im(α) (natural units)
        p_displacement = coherent_state.momentum_displacement()
        expected_p = np.sqrt(2) * np.imag(alpha)
        assert np.isclose(p_displacement, expected_p)
    
    def test_coherent_state_properties(self):
        """Test various coherent state properties."""
        alpha = 1.5
        coherent_state = CoherentState(alpha, n_max=20)
        
        # Coherent states are minimum uncertainty states
        # For ground state oscillator: Δx Δp = ℏ/2 (minimum possible)
        # This is preserved for coherent states
        
        # Test normalization
        assert np.isclose(np.sum(np.abs(coherent_state.amplitudes)**2), 1.0)
        
        # Test density matrix properties
        rho = coherent_state.density_matrix()
        
        # Should be Hermitian
        assert np.allclose(rho, rho.conj().T)
        
        # Should have unit trace
        assert np.isclose(np.trace(rho), 1.0)
        
        # Should be positive semi-definite (all eigenvalues >= 0)
        eigenvals = np.linalg.eigvals(rho)
        assert np.all(eigenvals.real >= -1e-12)  # Allow small numerical errors


class TestThermalState:
    """Test cases for ThermalState class."""
    
    def test_thermal_state_zero_temperature(self):
        """Test thermal state at T=0 (ground state)."""
        thermal_state = ThermalState(temperature=0.0, frequency=1.0, n_max=10)
        
        # At T=0, should be in ground state
        assert thermal_state.n_bar == 0.0
        assert thermal_state.probabilities[0] == 1.0
        
        for n in range(1, 11):
            assert thermal_state.probabilities[n] == 0.0
    
    def test_thermal_state_high_temperature(self):
        """Test thermal state at high temperature."""
        # High temperature: kT >> ℏω
        temperature = 10.0  # Much larger than ℏω = 1
        frequency = 1.0
        hbar = 1.0
        kb = 1.0
        
        thermal_state = ThermalState(temperature, frequency, hbar, kb, n_max=20)
        
        # At high temperature, n_bar should be approximately kT/(ℏω) - 1/2 ≈ kT/(ℏω)
        expected_n_bar = kb * temperature / (hbar * frequency)
        assert np.isclose(thermal_state.n_bar, expected_n_bar, rtol=0.1)
        
        # High temperature → classical equipartition
        assert thermal_state.n_bar > 1.0  # Should have significant excitation
    
    def test_thermal_state_density_matrix(self):
        """Test thermal state density matrix properties."""
        thermal_state = ThermalState(temperature=1.0, frequency=1.0, n_max=10)
        rho = thermal_state.density_matrix()
        
        # Should be diagonal (since it's a mixture of Fock states)
        off_diagonal = rho - np.diag(np.diag(rho))
        assert np.allclose(off_diagonal, 0.0)
        
        # Should be normalized (thermal distribution has infinite tail, so truncation 
        # at finite n_max introduces small normalization error - this is expected)
        assert np.isclose(np.trace(rho), 1.0, rtol=0.01)  # Allow 1% tolerance for truncation
        
        # Diagonal elements should match probabilities
        for n in range(len(thermal_state.probabilities)):
            assert np.isclose(rho[n, n], thermal_state.probabilities[n])
    
    def test_thermal_state_expectation_number(self):
        """Test thermal state photon number expectation."""
        temperature = 2.0
        frequency = 1.0
        thermal_state = ThermalState(temperature, frequency, n_max=15)
        
        # Two ways to calculate ⟨n⟩
        n_bar_direct = thermal_state.expectation_number()  # From n_bar formula
        
        n_bar_probabilities = sum(n * thermal_state.probabilities[n] 
                                 for n in range(len(thermal_state.probabilities)))
        
        assert np.isclose(n_bar_direct, n_bar_probabilities, rtol=0.01)
    
    def test_thermal_state_temperature_scaling(self):
        """Test thermal state behavior vs temperature."""
        frequency = 1.0
        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        n_bars = []
        for T in temperatures:
            thermal_state = ThermalState(T, frequency, n_max=20)
            n_bars.append(thermal_state.n_bar)
        
        # Mean photon number should increase monotonically with temperature
        for i in range(len(n_bars) - 1):
            assert n_bars[i+1] > n_bars[i]
        
        # At very low temperature, n_bar should be very small
        assert n_bars[0] < 0.1
        
        # At high temperature, n_bar should be large
        assert n_bars[-1] > 1.0