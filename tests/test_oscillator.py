"""Tests for quantum harmonic oscillator."""

import pytest
import numpy as np
from quantum_oscillator import HarmonicOscillator


class TestHarmonicOscillator:
    """Test cases for HarmonicOscillator class."""
    
    def test_initialization(self):
        """Test oscillator initialization."""
        oscillator = HarmonicOscillator(frequency=2.0, mass=1.5, hbar=1.0)
        
        assert oscillator.frequency == 2.0
        assert oscillator.mass == 1.5
        assert oscillator.hbar == 1.0
        
        # Test derived quantities
        expected_length_scale = np.sqrt(1.0 / (1.5 * 2.0))
        assert np.isclose(oscillator.length_scale, expected_length_scale)
        
        expected_energy_scale = 1.0 * 2.0
        assert oscillator.energy_scale == expected_energy_scale
    
    def test_energy_eigenvalues(self):
        """Test energy eigenvalue calculation."""
        oscillator = HarmonicOscillator(frequency=1.0, hbar=1.0)
        
        # Test first few energy levels: E_n = ℏω(n + 1/2)
        assert np.isclose(oscillator.energy_eigenvalue(0), 0.5)
        assert np.isclose(oscillator.energy_eigenvalue(1), 1.5)
        assert np.isclose(oscillator.energy_eigenvalue(2), 2.5)
        assert np.isclose(oscillator.energy_eigenvalue(10), 10.5)
        
        # Test error for negative n
        with pytest.raises(ValueError):
            oscillator.energy_eigenvalue(-1)
    
    def test_creation_annihilation_operators(self):
        """Test creation and annihilation operator matrices."""
        oscillator = HarmonicOscillator()
        n_max = 3
        
        # Creation operator a†
        a_dag = oscillator.creation_operator(n_max)
        
        # Check dimensions
        assert a_dag.shape == (n_max + 1, n_max + 1)
        
        # Check matrix elements: ⟨n+1|a†|n⟩ = √(n+1)
        assert np.isclose(a_dag[1, 0], np.sqrt(1))
        assert np.isclose(a_dag[2, 1], np.sqrt(2))
        assert np.isclose(a_dag[3, 2], np.sqrt(3))
        
        # Annihilation operator a
        a = oscillator.annihilation_operator(n_max)
        
        # Check matrix elements: ⟨n-1|a|n⟩ = √n
        assert np.isclose(a[0, 1], np.sqrt(1))
        assert np.isclose(a[1, 2], np.sqrt(2))
        assert np.isclose(a[2, 3], np.sqrt(3))
        
        # Check commutation relation: [a, a†] = I
        # Note: In finite basis, this only holds exactly in the valid subspace
        commutator = a @ a_dag - a_dag @ a
        expected_commutator = np.eye(n_max + 1)
        
        # Check commutation for all elements except boundary (where truncation effects appear)
        for i in range(n_max):
            for j in range(n_max):
                assert np.isclose(commutator[i, j], expected_commutator[i, j])
    
    def test_number_operator(self):
        """Test number operator."""
        oscillator = HarmonicOscillator()
        n_max = 5
        
        n_op = oscillator.number_operator(n_max)
        
        # Should be diagonal with eigenvalues 0, 1, 2, ..., n_max
        expected = np.diag([0, 1, 2, 3, 4, 5])
        assert np.allclose(n_op, expected)
    
    def test_hamiltonian(self):
        """Test Hamiltonian matrix."""
        oscillator = HarmonicOscillator(frequency=2.0, hbar=1.0)
        n_max = 3
        
        H = oscillator.hamiltonian(n_max)
        
        # Should be diagonal with eigenvalues ℏω(n + 1/2) = 2(n + 1/2)
        expected_eigenvalues = [1.0, 3.0, 5.0, 7.0]  # 2*(n + 0.5) for n = 0,1,2,3
        
        eigenvalues = np.diag(H)
        assert np.allclose(eigenvalues, expected_eigenvalues)
    
    def test_position_momentum_operators(self):
        """Test position and momentum operators."""
        oscillator = HarmonicOscillator(frequency=1.0, mass=1.0, hbar=1.0)
        n_max = 2
        
        x_op = oscillator.position_operator(n_max)
        p_op = oscillator.momentum_operator(n_max)
        
        # Check dimensions
        assert x_op.shape == (n_max + 1, n_max + 1)
        assert p_op.shape == (n_max + 1, n_max + 1)
        
        # Position operator should be Hermitian
        assert np.allclose(x_op, x_op.conj().T)
        
        # Momentum operator should also be Hermitian (p† = p)
        assert np.allclose(p_op, p_op.conj().T)
        
        # Check canonical commutation relation: [x, p] = iℏI (approximately)
        commutator = x_op @ p_op - p_op @ x_op
        expected = 1j * oscillator.hbar * np.eye(n_max + 1)
        
        # This should hold exactly for infinite-dimensional case
        # For truncated case, we check the leading elements
        assert np.isclose(commutator[0, 0], expected[0, 0])
        assert np.isclose(commutator[1, 1], expected[1, 1])
    
    def test_operator_relations(self):
        """Test relationships between operators."""
        oscillator = HarmonicOscillator(frequency=1.0, mass=1.0, hbar=1.0)
        n_max = 5
        
        a = oscillator.annihilation_operator(n_max)
        a_dag = oscillator.creation_operator(n_max)
        n_op = oscillator.number_operator(n_max)
        
        # Number operator relation: n = a†a
        n_from_operators = a_dag @ a
        
        # Should match except for the last row/column due to truncation
        for i in range(n_max):  # Exclude last element due to truncation effects
            for j in range(n_max):
                assert np.isclose(n_from_operators[i, j], n_op[i, j])
    
    def test_custom_units(self):
        """Test oscillator with custom units."""
        # Test with different frequency and mass
        freq = 3.0
        mass = 2.0
        hbar = 0.5
        
        oscillator = HarmonicOscillator(frequency=freq, mass=mass, hbar=hbar)
        
        # Energy scale should be ℏω
        expected_energy_scale = hbar * freq
        assert oscillator.energy_scale == expected_energy_scale
        
        # Length scale should be √(ℏ/mω)
        expected_length_scale = np.sqrt(hbar / (mass * freq))
        assert np.isclose(oscillator.length_scale, expected_length_scale)
        
        # Check energy eigenvalues
        E_0 = oscillator.energy_eigenvalue(0)
        expected_E_0 = hbar * freq * 0.5
        assert np.isclose(E_0, expected_E_0)