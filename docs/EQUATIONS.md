# Quantum Harmonic Oscillator: Governing Equations

This document presents the mathematical foundation of the quantum harmonic oscillator implementation in this toolkit.

## Core Hamiltonian

The quantum harmonic oscillator Hamiltonian in natural units (ℏ = m = ω = 1):

```
Ĥ = ℏω(â†â + 1/2)
```

Where:
- `ℏ` = reduced Planck constant
- `ω` = oscillator frequency  
- `â†` = creation operator
- `â` = annihilation operator

### Creation and Annihilation Operators

**Matrix Elements (Fock Basis):**
```
⟨n+1|â†|n⟩ = √(n+1)
⟨n-1|â|n⟩ = √n
```

**Commutation Relations:**
```
[â, â†] = 1
[â, â] = [â†, â†] = 0
```

## Energy Eigenvalues and Eigenstates

**Fock States (Number Eigenstates):**
```
Ĥ|n⟩ = En|n⟩
En = ℏω(n + 1/2)
```

For n = 0, 1, 2, ..., the energy spectrum is:
- E₀ = ℏω/2 (zero-point energy)
- E₁ = 3ℏω/2
- E₂ = 5ℏω/2
- ...

## Position and Momentum Operators

**In terms of ladder operators:**
```
x̂ = (ℓ/√2)(â + â†)
p̂ = i(√(ℏ/(2ℓ)))(â† - â)
```

Where `ℓ = √(ℏ/(mω))` is the characteristic length scale.

**Canonical Commutation Relation:**
```
[x̂, p̂] = iℏ
```

## Coherent States

**Definition:**
```
|α⟩ = e^(-|α|²/2) Σₙ (αⁿ/√(n!)) |n⟩
```

**Time Evolution:**
```
|α(t)⟩ = |α₀ e^(-iωt)⟩
```

**Classical Expectations:**
```
⟨x̂⟩ = √2 Re(α)
⟨p̂⟩ = √2 Im(α)
```

**Phase Space Dynamics:**
```
⟨x̂(t)⟩ = √2 |α| cos(ωt + φ)
⟨p̂(t)⟩ = -√2 |α| sin(ωt + φ)
```

Where φ = arg(α₀).

## Time Evolution

**Schrödinger Equation:**
```
iℏ ∂|ψ⟩/∂t = Ĥ|ψ⟩
```

**Unitary Evolution Operator:**
```
Û(t) = e^(-iĤt/ℏ)
```

**Solution:**
```
|ψ(t)⟩ = Û(t)|ψ(0)⟩
```

## Decoherence: Lindblad Master Equation

For open quantum systems coupled to environments:

```
∂ρ/∂t = -i[Ĥ, ρ]/ℏ + Σₖ γₖ 𝒟[L̂ₖ][ρ]
```

**Dissipator:**
```
𝒟[L̂][ρ] = L̂ρL̂† - (1/2){L̂†L̂, ρ}
```

### Amplitude Damping

**Lindblad Operator:**
```
L̂ = √γ â
```

**Physical Effect:**
- Energy dissipation: ⟨n⟩(t) = ⟨n⟩(0) e^(-γt)
- Purity decay: Tr(ρ²) decreases exponentially
- Quantum → Classical transition

### Phase Damping

**Lindblad Operator:**
```
L̂ = √γ_φ n̂ = √γ_φ â†â
```

**Physical Effect:**
- Pure dephasing (no energy loss)
- Coherence destruction in energy eigenbasis
- Off-diagonal density matrix elements decay

### Thermal Reservoir

**Lindblad Operators:**
```
L̂₁ = √γ(n̄ + 1) â     (cooling)
L̂₂ = √γn̄ â†           (heating)
```

Where `n̄ = 1/(e^(ℏω/kT) - 1)` is the thermal occupation number.

## Classical Limit

**Correspondence Principle:**

For large quantum numbers or coherent states with |α| >> 1:

```
⟨x̂(t)⟩ → x_cl(t) = A cos(ωt + φ)
⟨p̂(t)⟩ → p_cl(t) = -mωA sin(ωt + φ)
```

This demonstrates the quantum-to-classical transition fundamental to quantum decoherence studies.

## Uncertainty Relations

**Ground State (Minimum Uncertainty):**
```
Δx Δp = ℏ/2
```

**Coherent States:**
Maintain minimum uncertainty at all times:
```
Δx(t) Δp(t) = ℏ/2
```

**Fock States:**
```
Δx Δp = ℏ(n + 1/2)
```

Uncertainty grows with excitation number n.

---

These equations form the theoretical foundation for quantum oscillator dynamics, decoherence modeling, and the development of quantum error correction protocols.