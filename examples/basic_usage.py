#!/usr/bin/env python3
"""Basic usage examples for Quantum Oscillator Toolkit."""

import numpy as np
import matplotlib.pyplot as plt
from quantum_oscillator import HarmonicOscillator, CoherentState, FockState
from quantum_oscillator.evolution import TimeEvolution
from quantum_oscillator.decoherence import AmplitudeDamping, coherence_measures


def example_coherent_state_evolution():
    """Example: Time evolution of a coherent state."""
    print("=== Coherent State Evolution ===")
    
    # Create oscillator (frequency = 1.0, natural units)
    oscillator = HarmonicOscillator(frequency=1.0)
    
    # Create coherent state |α⟩ with α = 2.0
    alpha = 2.0 + 1.0j
    coherent_state = CoherentState(alpha, n_max=15)
    
    print(f"Initial coherent state: α = {alpha}")
    print(f"Initial ⟨n⟩ = {coherent_state.expectation_number():.3f}")
    
    # Time evolution
    evolution = TimeEvolution(oscillator)
    times = np.linspace(0, 2*np.pi, 100)
    
    # Track expectation values
    positions = []
    momenta = []
    
    for t in times:
        pos = evolution.expectation_position(coherent_state, t)
        mom = evolution.expectation_momentum(coherent_state, t)
        positions.append(pos)
        momenta.append(mom)
    
    # Plot phase space trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(positions, momenta, 'b-', linewidth=2, label='Quantum')
    
    # Compare with classical trajectory  
    x0 = np.sqrt(2) * np.real(alpha)
    p0 = np.sqrt(2) * np.imag(alpha)
    x_classical, p_classical = evolution.classical_trajectory(x0, p0, times)
    
    plt.plot(x_classical, p_classical, 'r--', linewidth=2, label='Classical')
    plt.xlabel('Position ⟨x⟩')
    plt.ylabel('Momentum ⟨p⟩')
    plt.title('Phase Space Trajectory: Coherent State')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('coherent_state_trajectory.png', dpi=150)
    plt.show()
    
    print("Phase space trajectory plotted!")


def example_fock_state_properties():
    """Example: Properties of Fock states."""
    print("\n=== Fock State Properties ===")
    
    oscillator = HarmonicOscillator(frequency=1.0)
    
    # Create several Fock states
    fock_states = [FockState(n, n_max=10) for n in range(5)]
    
    print("Fock state energies:")
    for n, state in enumerate(fock_states):
        energy = oscillator.energy_eigenvalue(n)
        print(f"|{n}⟩: E_{n} = {energy:.3f} ℏω")
    
    # Position and momentum uncertainties
    print("\nPosition/momentum uncertainties:")
    for n, state in enumerate(fock_states):
        x_op = oscillator.position_operator(10)
        p_op = oscillator.momentum_operator(10)
        rho = state.density_matrix()
        
        # ⟨x⟩, ⟨x²⟩
        x_mean = np.real(np.trace(x_op @ rho))
        x2_mean = np.real(np.trace(x_op @ x_op @ rho))
        x_uncertainty = np.sqrt(x2_mean - x_mean**2)
        
        # ⟨p⟩, ⟨p²⟩  
        p_mean = np.real(np.trace(p_op @ rho))
        p2_mean = np.real(np.trace(p_op @ p_op @ rho))
        p_uncertainty = np.sqrt(p2_mean - p_mean**2)
        
        # Uncertainty product
        uncertainty_product = x_uncertainty * p_uncertainty
        
        print(f"|{n}⟩: Δx = {x_uncertainty:.3f}, Δp = {p_uncertainty:.3f}, "
              f"ΔxΔp = {uncertainty_product:.3f}")


def example_decoherence():
    """Example: Decoherence of a coherent state."""
    print("\n=== Decoherence Example ===")
    
    oscillator = HarmonicOscillator(frequency=1.0)
    
    # Initial coherent state
    alpha = 2.0
    initial_state = CoherentState(alpha, n_max=15)
    initial_rho = initial_state.density_matrix()
    
    print(f"Initial coherent state: α = {alpha}")
    initial_coherence = coherence_measures(initial_rho)
    print(f"Initial purity: {initial_coherence['purity']:.3f}")
    
    # Apply amplitude damping
    damping = AmplitudeDamping(oscillator, damping_rate=0.1)
    
    times = np.linspace(0, 5, 50)
    purities = []
    photon_numbers = []
    
    for t in times:
        # Evolve with decoherence
        evolved_rho = damping.evolve_density_matrix(initial_rho, t)
        
        # Calculate measures
        coherence = coherence_measures(evolved_rho)
        purities.append(coherence['purity'])
        
        # Photon number expectation
        n_op = oscillator.number_operator(15)
        n_mean = np.real(np.trace(n_op @ evolved_rho))
        photon_numbers.append(n_mean)
    
    # Plot decoherence
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    ax1.plot(times, purities, 'b-', linewidth=2)
    ax1.set_ylabel('Purity')
    ax1.set_title('Decoherence: Amplitude Damping')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, photon_numbers, 'r-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('⟨n⟩')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decoherence_evolution.png', dpi=150)
    plt.show()
    
    print("Decoherence evolution plotted!")


def example_wigner_function():
    """Example: Wigner function for quantum states (simplified).""" 
    print("\n=== Wigner Function Visualization ===")
    
    # This is a simplified visualization - full Wigner function requires more computation
    oscillator = HarmonicOscillator(frequency=1.0)
    
    # Create states
    ground_state = FockState(0, n_max=10)
    excited_state = FockState(1, n_max=10) 
    coherent_state = CoherentState(1.5, n_max=10)
    
    states = [
        (ground_state, "Ground State |0⟩"),
        (excited_state, "First Excited |1⟩"), 
        (coherent_state, "Coherent |α=1.5⟩")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (state, title) in enumerate(states):
        ax = axes[i]
        
        # Fock state probabilities (bar plot approximation to Wigner function)
        n_values = range(10)
        probabilities = [state.probability(n) for n in n_values]
        
        ax.bar(n_values, probabilities, alpha=0.7)
        ax.set_xlabel('Fock State |n⟩')
        ax.set_ylabel('Probability')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantum_states_comparison.png', dpi=150)
    plt.show()
    
    print("Quantum states comparison plotted!")


if __name__ == "__main__":
    # Run all examples
    example_coherent_state_evolution()
    example_fock_state_properties()
    example_decoherence()
    example_wigner_function()
    
    print("\n✅ All examples completed successfully!")
    print("Check generated PNG files for visualizations.")