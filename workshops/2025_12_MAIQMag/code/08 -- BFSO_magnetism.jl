# Using SU(5) to model Ba₂FeSi₂O₇ with spin s=2
#
# BFSO has strong easy-plane anisotropy. The single-ion physics is well modeled
# using the theory of SU(5) coherent states. This follows previous studies:
#
# - S.-H. Do et al., "Decay and renormalization of a longitudinal mode...,"
#   [Nature Communications **12**
#   (2021)](https://doi.org/10.1038/s41467-021-25591-7).
# - M. Lee et al., "Field-induced spin level crossings...," [PRB **107**
#   (2023)](https://doi.org/10.1103/PhysRevB.107.144427).
# - S.-H. Do et al., "Understanding temperature-dependent SU(3) spin
#   dynamics...," [npj quantum materials **5**
#   (2023)](https://doi.org/10.1038/s41535-022-00526-7).

# # 1. Anisotropies and large spins 
#
# First we'll consider a cartoon picture of the single-ion physics.

using Sunny, GLMakie, LinearAlgebra, Statistics

S = spin_matrices(2)  # Returns a vector of Sx, Sy, Sz
Sx, Sy, Sz = S        # Julia's "unpacking" syntax

# In the usual basis, this operator is a diagonal matrix

H_SI = Sz^2

# This quantum spin state is one of the eigenmodes

Z = [0., 0, 1 + 0im, 0, 0]

# We can inspect expectation values

expectation(op, Z) = real(Z' * op * Z)

# This is a non-magnetic state

expectation(Sx, Z)
expectation(Sy, Z)
expectation(Sz, Z)

# But quadrupole moments are not trivial

expectation(2Sz^2 - Sx^2 - Sy^2, Z) 

# We'll build a model consisting of just single-ion anisotropy

latvecs = lattice_vectors(1, 1, 1.2, 90, 90, 90)
positions = [[0, 0, 0]]
crystal = Crystal(latvecs, positions)
view_crystal(crystal)

dims = (6, 6, 2)
sys = System(crystal, [1 => Moment(s=2, g=2)], :SUN; dims, seed=1)
set_onsite_coupling!(sys, +Sz^2, 1) # Set the anisotropy term

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

# Closer inspection shows that the spins are effectively vanishing.
## EXERCISE: See what happens in `:dipole` mode.
## EXERCISE: See what happens when you change the sign of the anisotropy. 

norm(sys.dipoles[1,1,1,1])

# Verify the coherent state is equal to `Z=(0, 0, 1, 0, 0)`, i.e. the spin state
# |s=2, m=0⟩

sys.coherents[1,1,1,1]
norm.(sys.coherents[1,1,1,1])


# BFSO is more complicated, but has a predominantly easy-plane (hard-axis)
# character.

# # 2. BFSO Hamiltonian specification

units = Units(:K, :angstrom)
a = 8.3194
c = 5.336
latvecs = lattice_vectors(a, a, c, 90, 90, 90)
positions = [[0, 0, 0]]
spacegroup = 113    # Want to use the space group for original lattice, of which the Fe ions form a subcrystal
crystal = Crystal(latvecs, positions, spacegroup; types=["Fe"])
view_crystal(crystal)

dims = (6, 6, 2)
sys = System(crystal, [1 => Moment(s=2, g=1.93)], :SUN; dims)

A = 1.16
C = -1.74
D = 28.65
Sx, Sy, Sz = spin_matrices(2)
H_SI = D*(Sz)^2 + A*((Sx)^4 + (Sy)^4) + C*(Sz)^4
set_onsite_coupling!(sys, H_SI, 1)

# One can also work in Stevens operators

print_stevens_expansion(H_SI)

O = stevens_matrices(2)

display((2213/420)O[2, 0] - 0.02486O[4, 0] + (29/100)O[4, 4] + 15311/250*I)
display(H_SI)

# Next define exchange interactions

bond1 = Bond(1, 2, [0, 0, 0])
bond2 = Bond(1, 1, [1, 0, 0])
bond3 = Bond(1, 1, [0, 0, 1])

J = 1.028 * units.K
J′ = 0.1J
set_exchange!(sys, J, bond1)
set_exchange!(sys, J′, bond2)
set_exchange!(sys, J′, bond3)

# We have now completely specified our Hamiltonian. Let's examine the zero-field
# ground state (staggered XY-ordering in the plane).

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

## EXERCISE: Examine the `dipoles` and `coherents` fields. 
## EXERCISE: Use `set_field!` to see how the ground state develops with applied field.

# # 3. M vs. H

# Helper function to calculate the magnetization per site.

function magnetization(sys)
    return norm(mean(magnetic_moment(sys, site) for site in eachsite(sys)))
end

magnetization(sys)

# Helper function to calculate the relevant order parameter, the staggered
# magnetization in the plane. 

function order_parameter(sys)
    xy1 = [1/√2, 1/√2, 0]   # Unit vector in the (1, -1, 0) direction
    xy2 = [-1/√2, 1/√2, 0]  # Unit vector in the (1, 1, 0) direction
    M_xy1 = 0.0
    M_xy2 = 0.0
    for site in eachsite(sys)
        sublattice = (-1)^(site[4]) * (-1)^(site[3])
        M_xy1 = sublattice * (magnetic_moment(sys, site) ⋅ xy1)
        M_xy2 = sublattice * (magnetic_moment(sys, site) ⋅ xy2)
    end
    return max(abs(M_xy1), abs(M_xy2))
end

order_parameter(sys)

# Iterate through a list of applied field values, reoptimizing the spin
# configuration each time.

Hs = range(0.0, 55.0, 50)
Ms = Float64[]
OPs = Float64[]
for H in Hs
    set_field!(sys, (0, 0, H*units.T))
    minimize_energy!(sys)
    push!(Ms, magnetization(sys))
    push!(OPs, order_parameter(sys))
end

fig = Figure(size=(1200,400))
lines(fig[1,1], Hs, Ms; axis=(xlabel="H", ylabel="M"), color=:red)
lines(fig[1,2], Hs, OPs; axis=(xlabel="H", ylabel="Staggered XY Magnetization"), color=:red)
fig
