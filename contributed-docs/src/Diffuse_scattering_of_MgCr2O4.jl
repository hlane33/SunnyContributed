# # Diffuse scattering of MgCr₂O₄

# This tutorial introduces the self-consistent Gaussian approximation (SCGA) to
# simulate the diffuse scattering data of materials above their ordering
# temperature, where spin fluctuations are approximated to be Gaussian. This method
# relaxes the spin length constraint on each site, enforcing spin normalization on 
# average `` \langle \boldsymbol{S} \rangle = s``. SCGA is exact in 
# the limit of large number of spin components, ``n \to \infty``, but may fail at low
# temperatures or close to magnetic order for finite dimensional spins.

# Diffuse scattering contains rich information about the spin correlations in 
# interacting spin systems above the magnetic ordering transition which may be 
# modeled to parameterize a spin Hamiltonian. It should be emphasized that this 
# approach is not dependent on the selection of a spin state which is a local
# minimum of the classical energy, making SCGA an efficient tool to determine 
# initial parameters ahead of a more comprehensive spin wave fitting.  

# As an example, we consider the frustrated pyrochlore antiferromagnet
# MgCr₂O₄, which is proxmiate to a spiral spin liquid phase. We reproduce the 
# diffuse the diffuse scattering data of [Bai et al., Phys. Rev. Lett. **122**,
# 097201 (2019)](https://doi.org/10.1103/PhysRevLett.122.097201).

# The first step is to load the required packages and set the units.

using Sunny, GLMakie, LinearAlgebra
units = Units(:meV,:angstrom)

# ### Creating the crystal structure

# Next, we create the MgCr₂O₄ unit cell by specifying the lattice vectors,
# atomic positions, atom types and space group.

latvecs = lattice_vectors(8.3342, 8.3342, 8.3342, 90, 90, 90)
positions = [[0.1250, 0.1250, 0.1250],
            [0.5000, 0.5000, 0.5000],
            [0.2607, 0.2607, 0.2607]]
types = ["Mg","Cr","O"]
spacegroup = 227
cryst = Crystal(latvecs, positions, spacegroup; types)

# We can focus solely on the magnetic chromium ions here, so we
# create a subcrystal.

subcryst = subcrystal(cryst, "Cr");
view_crystal(subcryst);

# ### Creating the spin system

# We now have just one atom type in our crystal, so we set the [`Moment`](@ref) 
# using ``s=3/2`` and ``g=2`` for our chromium ions and create the [`System`](@ref). 
# Note that [`SCGA`](@ref) currenty only supports `:dipole` and `:dipole_uncorrected`.
# The exchange constants are those determined by 
# [Bai et al](https://doi.org/10.1103/PhysRevLett.122.097201).

spininfos = [1 => Moment(; s=3/2, g=2)]  
sys = System(subcryst, spininfos, :dipole); 
J1 = 3.27  
J_mgcro = [1.00,0.0815,0.1050,0.0085]*J1; 
set_exchange!(sys, J_mgcro[1], Bond(1, 2, [0,0,0])) 
set_exchange!(sys, J_mgcro[2], Bond(1, 7, [0,0,0]))  
set_exchange!(sys, J_mgcro[3], Bond(1, 3, [1,0,0])) 
set_exchange!(sys, J_mgcro[4], Bond(1, 3, [0,0,0]))

# We now include an isotropic [`FormFactor`](@ref) specific to Cr³⁺. 
# The use of [`ssf_perp`](@ref) specifies that we are measuring the projection 
# of the spin structure factor,  ``\mathcal{S}(𝐪,ω)``, perpendicular to the 
# direction of momentum transfer.

formfactors = [1 => FormFactor("Cr3")]
measure = ssf_perp(sys; formfactors);

# The self-consistent Gaussian approximation constrains the average spin magnitude
# to be equal to ``s``, which in our case is ``3/2``.
# [`SCGA`](@ref) takes in a `kT` value and assumes a classical Boltzmann 
# distribution at this temperature. An optional key word argument, `dq`, 
# sets the grid spacing for the momentum integral to determine the Lagrange
# multipler.  

# The transition temperature of MgCr₂O₄ is ≈ 13 K. Hence, Bai et al. measured 
# inelastic neutron scattering data at 20 K, within the correlated 
# paramagnetic regime.

kT = 20units.K
scga =  SCGA(sys; measure, kT, dq= 0.1);

# The output of [`SCGA`](@ref) is an [`intensities_static`](@ref) object.
# We plot intensity for a given reciprocal-space plane to allow for comparison with
# the experimental data from [Bai et al](https://doi.org/10.1103/PhysRevLett.122.097201). 

grid = q_space_grid(cryst, [1,0,0], range(-4,4,200), [0,1,0], range(-4,4,200))
res = intensities_static(scga, grid)
plot_intensities(res)

