#!/usr/bin/env python3
"""
Visualize the governing equations and dynamics of quantum harmonic oscillator systems.

This script generates plots showing:
1. Energy spectrum and wavefunctions
2. Classical vs quantum trajectories  
3. Coherent state time evolution
4. Decoherence dynamics
5. Phase space portraits
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_oscillator import HarmonicOscillator, CoherentState, FockState
from quantum_oscillator.evolution import TimeEvolution
from quantum_oscillator.decoherence import AmplitudeDamping, PhaseDamping


def plot_energy_spectrum():
    """Plot energy eigenvalues and demonstrate E_n = ℏω(n + 1/2)."""
    oscillator = HarmonicOscillator(frequency=1.0, hbar=1.0)
    
    n_levels = 8
    n_values = np.arange(n_levels)
    energies = [oscillator.energy_eigenvalue(n) for n in n_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy level diagram
    ax1.hlines(energies, 0, 1, colors='blue', linewidth=3)
    for i, E in enumerate(energies):
        ax1.text(1.1, E, f'|{i}⟩, E = {E:.1f}ℏω', va='center', fontsize=10)
    
    ax1.set_xlim(0, 2.5)
    ax1.set_ylabel('Energy (ℏω)')
    ax1.set_title('Energy Spectrum: Eₙ = ℏω(n + ½)')
    ax1.set_xticks([])
    ax1.grid(True, alpha=0.3)
    
    # Energy vs quantum number
    ax2.plot(n_values, energies, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Quantum Number n')
    ax2.set_ylabel('Energy Eₙ (ℏω)')
    ax2.set_title('Linear Energy Spacing')
    ax2.grid(True, alpha=0.3)
    
    # Add equation text
    ax2.text(0.7, 0.8, r'$E_n = \hbar\omega(n + \frac{1}{2})$', 
             transform=ax2.transAxes, fontsize=16, 
             bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig('energy_spectrum.png', dpi=150)
    plt.show()


def plot_classical_quantum_trajectories():
    """Compare classical and quantum trajectories in phase space."""
    oscillator = HarmonicOscillator(frequency=1.0)
    evolution = TimeEvolution(oscillator)
    
    # Create coherent states with different amplitudes
    alphas = [1.0, 2.0, 3.0]
    colors = ['red', 'green', 'blue']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (alpha, color) in enumerate(zip(alphas, colors)):
        ax = axes[i]
        
        coherent_state = CoherentState(alpha, n_max=20)
        
        # Time for one complete oscillation
        times = np.linspace(0, 2*np.pi, 100)
        
        # Quantum expectation values
        q_positions = [evolution.expectation_position(coherent_state, t) for t in times]
        q_momenta = [evolution.expectation_momentum(coherent_state, t) for t in times]
        
        # Classical trajectory
        x0 = np.sqrt(2) * alpha  # Initial position
        p0 = 0.0                 # Initial momentum
        c_positions, c_momenta = evolution.classical_trajectory(x0, p0, times)
        
        # Plot trajectories
        ax.plot(q_positions, q_momenta, color=color, linewidth=2.5, 
                label=f'Quantum |α={alpha}⟩', alpha=0.8)
        ax.plot(c_positions, c_momenta, '--', color=color, linewidth=2, 
                label=f'Classical', alpha=0.6)
        
        # Mark starting point
        ax.plot(q_positions[0], q_momenta[0], 'o', color=color, markersize=8)
        
        ax.set_xlabel('Position ⟨x⟩')
        ax.set_ylabel('Momentum ⟨p⟩')
        ax.set_title(f'Phase Space: α = {alpha}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle('Classical-Quantum Correspondence: ⟨x(t)⟩ = A cos(ωt), ⟨p(t)⟩ = -A sin(ωt)', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('classical_quantum_trajectories.png', dpi=150)
    plt.show()


def plot_coherent_state_evolution():
    """Show time evolution of coherent state parameter α(t) = α₀ e^(-iωt).""" 
    oscillator = HarmonicOscillator(frequency=2.0)  # Higher frequency for faster oscillation
    evolution = TimeEvolution(oscillator)
    
    # Complex initial coherent state
    alpha_0 = 2.0 + 1.0j
    
    times = np.linspace(0, 2*np.pi/2.0, 200)  # One period at ω=2
    
    # Coherent state parameter evolution
    alphas_t = evolution.coherent_state_trajectory(alpha_0, times)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Complex plane trajectory
    ax1.plot(np.real(alphas_t), np.imag(alphas_t), 'b-', linewidth=2)
    ax1.plot(np.real(alpha_0), np.imag(alpha_0), 'ro', markersize=10, label='t=0')
    ax1.plot(np.real(alphas_t[-1]), np.imag(alphas_t[-1]), 'go', markersize=10, label='t=T')
    ax1.set_xlabel('Re(α)')
    ax1.set_ylabel('Im(α)')
    ax1.set_title('Complex Plane: α(t) = α₀ e^(-iωt)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Magnitude vs time (should be constant)
    ax2.plot(times, np.abs(alphas_t), 'r-', linewidth=2)
    ax2.axhline(abs(alpha_0), color='black', linestyle='--', alpha=0.5, label='|α₀|')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('|α(t)|')
    ax2.set_title('Magnitude Conservation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Phase vs time (linear decrease)
    phases = np.angle(alphas_t)
    ax3.plot(times, np.unwrap(phases), 'g-', linewidth=2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('arg(α(t))')
    ax3.set_title('Phase Evolution: φ(t) = φ₀ - ωt')
    ax3.grid(True, alpha=0.3)
    
    # Real and imaginary parts
    ax4.plot(times, np.real(alphas_t), 'b-', linewidth=2, label='Re(α)')
    ax4.plot(times, np.imag(alphas_t), 'r-', linewidth=2, label='Im(α)')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('α components')
    ax4.set_title('Sinusoidal Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('coherent_state_evolution.png', dpi=150)
    plt.show()


def plot_decoherence_comparison():
    """Compare different types of decoherence (amplitude vs phase damping)."""
    oscillator = HarmonicOscillator(frequency=1.0)
    
    # Initial coherent state
    alpha = 2.0
    initial_state = CoherentState(alpha, n_max=20)
    initial_rho = initial_state.density_matrix()
    
    # Different decoherence models
    amp_damping = AmplitudeDamping(oscillator, damping_rate=0.2)
    phase_damping = PhaseDamping(oscillator, dephasing_rate=0.2)
    
    times = np.linspace(0, 5, 100)
    
    # Track different quantities
    amp_purity = []
    amp_energy = []
    phase_purity = []
    phase_energy = []
    
    n_op = oscillator.number_operator(20)
    
    for t in times:
        # Amplitude damping evolution
        rho_amp = amp_damping.evolve_density_matrix(initial_rho, t)
        amp_purity.append(np.real(np.trace(rho_amp @ rho_amp)))
        amp_energy.append(np.real(np.trace(n_op @ rho_amp)))
        
        # Phase damping evolution
        rho_phase = phase_damping.evolve_density_matrix(initial_rho, t)
        phase_purity.append(np.real(np.trace(rho_phase @ rho_phase)))
        phase_energy.append(np.real(np.trace(n_op @ rho_phase)))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Purity comparison
    ax1.plot(times, amp_purity, 'r-', linewidth=2.5, label='Amplitude Damping')
    ax1.plot(times, phase_purity, 'b-', linewidth=2.5, label='Phase Damping')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Purity Tr(ρ²)')
    ax1.set_title('Purity Loss: Different Mechanisms')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Energy comparison
    ax2.plot(times, amp_energy, 'r-', linewidth=2.5, label='Amplitude Damping')
    ax2.plot(times, phase_energy, 'b-', linewidth=2.5, label='Phase Damping')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy ⟨n⟩')
    ax2.set_title('Energy Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Lindblad equation visualization
    ax3.text(0.05, 0.9, r'Lindblad Master Equation:', fontsize=14, weight='bold',
             transform=ax3.transAxes)
    ax3.text(0.05, 0.8, r'$\frac{\partial\rho}{\partial t} = -i[H,\rho] + \sum_k \gamma_k \mathcal{D}[L_k][\rho]$',
             fontsize=12, transform=ax3.transAxes)
    ax3.text(0.05, 0.65, r'Dissipator:', fontsize=12, weight='bold', transform=ax3.transAxes)
    ax3.text(0.05, 0.55, r'$\mathcal{D}[L][\rho] = L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho\}$',
             fontsize=12, transform=ax3.transAxes)
    ax3.text(0.05, 0.4, r'Amplitude Damping: $L = \sqrt{\gamma} \hat{a}$',
             fontsize=11, color='red', transform=ax3.transAxes)
    ax3.text(0.05, 0.3, r'Phase Damping: $L = \sqrt{\gamma_\phi} \hat{n}$',
             fontsize=11, color='blue', transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Physical interpretation
    ax4.text(0.05, 0.9, r'Physical Effects:', fontsize=14, weight='bold',
             transform=ax4.transAxes)
    ax4.text(0.05, 0.75, r'Amplitude Damping:', fontsize=12, color='red', weight='bold',
             transform=ax4.transAxes)
    ax4.text(0.05, 0.65, r'• Energy dissipation', fontsize=11, transform=ax4.transAxes)
    ax4.text(0.05, 0.55, r'• Purity loss', fontsize=11, transform=ax4.transAxes)
    ax4.text(0.05, 0.45, r'• Cooling toward ground state', fontsize=11, transform=ax4.transAxes)
    ax4.text(0.05, 0.3, r'Phase Damping:', fontsize=12, color='blue', weight='bold',
             transform=ax4.transAxes)
    ax4.text(0.05, 0.2, r'• Pure dephasing (no energy loss)', fontsize=11, transform=ax4.transAxes)
    ax4.text(0.05, 0.1, r'• Coherence destruction', fontsize=11, transform=ax4.transAxes)
    ax4.text(0.05, 0.0, r'• Off-diagonal decay', fontsize=11, transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('decoherence_comparison.png', dpi=150)
    plt.show()


def plot_uncertainty_relations():
    """Visualize uncertainty relations for different quantum states."""
    oscillator = HarmonicOscillator(frequency=1.0)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Fock states uncertainty scaling
    n_values = np.arange(10)
    uncertainties = []
    
    for n in n_values:
        fock_state = FockState(n, n_max=15)
        x_op = oscillator.position_operator(15)
        p_op = oscillator.momentum_operator(15)
        rho = fock_state.density_matrix()
        
        # Calculate uncertainties
        x_mean = np.real(np.trace(x_op @ rho))
        x2_mean = np.real(np.trace(x_op @ x_op @ rho))
        delta_x = np.sqrt(x2_mean - x_mean**2)
        
        p_mean = np.real(np.trace(p_op @ rho))
        p2_mean = np.real(np.trace(p_op @ p_op @ rho))
        delta_p = np.sqrt(p2_mean - p_mean**2)
        
        uncertainties.append(delta_x * delta_p)
    
    ax1.plot(n_values, uncertainties, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(0.5, color='red', linestyle='--', label='ℏ/2 (minimum)')
    ax1.set_xlabel('Fock State |n⟩')
    ax1.set_ylabel('ΔxΔp')
    ax1.set_title('Uncertainty Product: Fock States')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Coherent state uncertainty (constant)
    alpha_values = np.linspace(0, 3, 50)
    coherent_uncertainties = []
    
    for alpha in alpha_values:
        coherent_state = CoherentState(alpha, n_max=20)
        # Use consistent dimension for operators
        x_op_coherent = oscillator.position_operator(20)
        p_op_coherent = oscillator.momentum_operator(20)
        rho = coherent_state.density_matrix()
        
        x_mean = np.real(np.trace(x_op_coherent @ rho))
        x2_mean = np.real(np.trace(x_op_coherent @ x_op_coherent @ rho))
        delta_x = np.sqrt(x2_mean - x_mean**2)
        
        p_mean = np.real(np.trace(p_op_coherent @ rho))
        p2_mean = np.real(np.trace(p_op_coherent @ p_op_coherent @ rho))
        delta_p = np.sqrt(p2_mean - p_mean**2)
        
        coherent_uncertainties.append(delta_x * delta_p)
    
    ax2.plot(alpha_values, coherent_uncertainties, 'g-', linewidth=2.5)
    ax2.axhline(0.5, color='red', linestyle='--', label='ℏ/2 (minimum)')
    ax2.set_xlabel('Coherent State |α|')
    ax2.set_ylabel('ΔxΔp')
    ax2.set_title('Minimum Uncertainty: Coherent States')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Equation display
    ax3.text(0.05, 0.9, r'Heisenberg Uncertainty Principle:', fontsize=14, weight='bold',
             transform=ax3.transAxes)
    ax3.text(0.05, 0.75, r'$\Delta x \Delta p \geq \frac{\hbar}{2}$',
             fontsize=16, transform=ax3.transAxes)
    ax3.text(0.05, 0.55, r'For Quantum Harmonic Oscillator:', fontsize=12, weight='bold',
             transform=ax3.transAxes)
    ax3.text(0.05, 0.4, r'Fock States: $\Delta x \Delta p = \hbar(n + \frac{1}{2})$',
             fontsize=12, color='blue', transform=ax3.transAxes)
    ax3.text(0.05, 0.25, r'Coherent States: $\Delta x \Delta p = \frac{\hbar}{2}$ (minimum)',
             fontsize=12, color='green', transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Physical interpretation
    ax4.text(0.05, 0.9, r'Physical Meaning:', fontsize=14, weight='bold',
             transform=ax4.transAxes)
    ax4.text(0.05, 0.75, r'• Fundamental quantum limit on measurement precision',
             fontsize=11, transform=ax4.transAxes)
    ax4.text(0.05, 0.65, r'• Coherent states are "most classical"',
             fontsize=11, transform=ax4.transAxes)
    ax4.text(0.05, 0.55, r'• Higher Fock states have larger uncertainty',
             fontsize=11, transform=ax4.transAxes)
    ax4.text(0.05, 0.45, r'• Decoherence can increase uncertainty',
             fontsize=11, transform=ax4.transAxes)
    ax4.text(0.05, 0.3, r'Quantum Error Correction Goal:', fontsize=12, color='red', weight='bold',
             transform=ax4.transAxes)
    ax4.text(0.05, 0.2, r'Maintain minimum uncertainty states', fontsize=11, color='red',
             transform=ax4.transAxes)
    ax4.text(0.05, 0.1, r'Combat decoherence-induced spreading', fontsize=11, color='red',
             transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('uncertainty_relations.png', dpi=150)
    plt.show()


def main():
    """Generate all equation visualization plots."""
    print("Generating quantum harmonic oscillator equation visualizations...")
    print()
    
    print("1. Energy spectrum and eigenvalues...")
    plot_energy_spectrum()
    
    print("2. Classical vs quantum trajectories...")
    plot_classical_quantum_trajectories()
    
    print("3. Coherent state time evolution...")
    plot_coherent_state_evolution()
    
    print("4. Decoherence mechanisms comparison...")
    plot_decoherence_comparison()
    
    print("5. Uncertainty relations...")
    plot_uncertainty_relations()
    
    print("\n✅ All equation plots generated successfully!")
    print("Files created:")
    print("  - energy_spectrum.png")
    print("  - classical_quantum_trajectories.png") 
    print("  - coherent_state_evolution.png")
    print("  - decoherence_comparison.png")
    print("  - uncertainty_relations.png")


if __name__ == "__main__":
    main()