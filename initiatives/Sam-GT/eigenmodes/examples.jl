# # Eigenmode Viewer
#
# This script demonstrates how to use the eigenmode viewer:

cd("../eigenmodes")#hide
include("eigenmode_viewer.jl")
nothing#hide

# To launch the interactive eigenmode viewer, use this command: `interact_eigenmodes(swt, qpath)`

# The same script also provides a function `get_eigenmodes`, which gives programmatic access to the
# LSWT eigenmodes.
# We will demonstrate how to use this with the basic example of a spin-1 antiferromagnet with easy-xy-plane anisotropy:

function example_afm()
  a = b = 8.539
  c = 5.2414
  latvecs = lattice_vectors(a, b, c, 90, 90, 90)
  crystal = Crystal(latvecs,[[0.,0,0]],1)
  latsize = (2,1,1)
  sys = System(crystal, [1 => Moment(;s=1, g=2)], :SUN; seed=5,dims = latsize)
  set_exchange!(sys, 0.85,  Bond(1, 1, [1,0,0]))   # J1
  set_onsite_coupling!(sys, S -> 0.3 * S[3]^2,1)

  #sys.dipoles[1] = SVector{3}([0,0,1])
  #sys.dipoles[2] = SVector{3}([0,0,-1])
  randomize_spins!(sys)
  minimize_energy!(sys)

  swt = SpinWaveTheory(sys;measure = ssf_perp(sys))
  qpath = q_space_path(crystal,[[0,0,0], [1,1,0]], 800)
  swt, qpath
end

example_afm_swt, qpath = example_afm()

display(example_afm_swt.sys)

# Now, we can run the LSWT eigenanalysis at a particular wavevector, which gives us
# several pieces of data:

particular_wavevector = qpath.qs[35]
energies,T = excitations(example_afm_swt, particular_wavevector)
Z_cos, Z_sin = get_eigenmodes(example_afm_swt, particular_wavevector)
nothing#hide

# The bogoliubov matrix diagonalizing $H$:
@show size(T);

# The eigenenergies:
@show size(energies);

# The corresponding modeshapes. Because the coherent states are inherently
# complex-valued, and the spin wave can also be elliptically polarized on top
# of that, we report separately the in-phase and quadrature "cos" and "sin"
# components of the spin wave polarization:
@show size(Z_cos);
@show size(Z_sin);

# Each element of `Z_cos` and `Z_sin` is a list of modeshapes, where each modeshape
# is an N × [number of sublattices] matrix containing the perturbation of the
# ground state SU(N) coherent state associated with each sublattice.
# There is one such in-phase modeshape, and one such quadrature modeshape for each
# band (eigenmode) in the band structure. We can plot a vizualization of the
# dipole sector of the displacement for a particular eigenmode as follows:

band = 3
modeshapes = (Z_cos[band],Z_sin[band])
k = Observable([0,0,0])
plot_eigenmode(Observable(modeshapes),example_afm_swt;k)

# By default, this is a static view, so each spin is only displaying one particular
# phase of its full cycle. To get the full picture, we need to look at a few times:

f = Figure(); ax = LScene(f[1,1],show_axis = false);
for t = [0,π/2,π,3π/2]
  plot_eigenmode!(ax,Observable(modeshapes),example_afm_swt; t = Observable(t),k)
end
f

# The plot can be updated dynamically in the usual [GLMakie](https://docs.makie.org/stable/) way by updating the `Observable` values passed to `plot_eigenmode`, e.g. `t[] = 0.1; notify(t)`.
#
# Better yet, **the full eigenmode viewer can be started with this command**: `interact_eigenmodes(swt, qs)`.
# Below are two more example systems that can be used to try out `interact_eigenmodes`:

function example_eigenmodes()
  cryst = Sunny.cubic_crystal()
  sys = System(cryst, [1 => Moment(;s=1/2,g=1)], :SUN;dims = (1,2,1))
  set_field!(sys,[0,0,0.5]) # Field along Z
  set_exchange!(sys,-1.,Bond(1,1,[0,1,0])) # Strong Ferromagnetic J
  randomize_spins!(sys)
  minimize_energy!(sys)
  minimize_energy!(sys)
  minimize_energy!(sys)

  swt = SpinWaveTheory(sys;measure = ssf_perp(sys))
  qpath = q_space_path(cryst,[[0,0,0], [0,1,0]], 40)
  get_eigenmodes(swt,qpath.qs[3])
  interact_eigenmodes(swt, qpath)
end


function example_fei2()
  a = b = 4.05012  
  c = 6.75214     
  latvecs = lattice_vectors(a, b, c, 90, 90, 120) 
                                                 
  positions = [[0, 0, 0], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]  
                                                            
  types = ["Fe", "I", "I"]
  FeI2 = Crystal(latvecs, positions; types)
  cryst = subcrystal(FeI2, "Fe")
  sys = System(cryst, [1 => Moment(;s=1, g=2)], :SUN, seed=2;dims=(4,4,4))
  J1pm   = -0.236
  J1pmpm = -0.161
  J1zpm  = -0.261
  J2pm   = 0.026
  J3pm   = 0.166
  J′0pm  = 0.037
  J′1pm  = 0.013
  J′2apm = 0.068

  J1zz   = -0.236
  J2zz   = 0.113
  J3zz   = 0.211
  J′0zz  = -0.036
  J′1zz  = 0.051
  J′2azz = 0.073

  J1xx = J1pm + J1pmpm
  J1yy = J1pm - J1pmpm
  J1yz = J1zpm

  set_exchange!(sys, [J1xx   0.0    0.0;
                      0.0    J1yy   J1yz;
                      0.0    J1yz   J1zz], Bond(1,1,[1,0,0]))
  set_exchange!(sys, [J2pm   0.0    0.0;
                      0.0    J2pm   0.0;
                      0.0    0.0    J2zz], Bond(1,1,[1,2,0]))
  set_exchange!(sys, [J3pm   0.0    0.0;
                      0.0    J3pm   0.0;
                      0.0    0.0    J3zz], Bond(1,1,[2,0,0]))
  set_exchange!(sys, [J′0pm  0.0    0.0;
                      0.0    J′0pm  0.0;
                      0.0    0.0    J′0zz], Bond(1,1,[0,0,1]))
  set_exchange!(sys, [J′1pm  0.0    0.0;
                      0.0    J′1pm  0.0;
                      0.0    0.0    J′1zz], Bond(1,1,[1,0,1]))
  set_exchange!(sys, [J′2apm 0.0    0.0;
                      0.0    J′2apm 0.0;
                      0.0    0.0    J′2azz], Bond(1,1,[1,2,1]))

  D = 2.165
  S = spin_operators(sys, 1)
  set_onsite_coupling!(sys, -D*S[3]^2, 1)

  randomize_spins!(sys)
  minimize_energy!(sys);

  sys_min = reshape_supercell(sys, [1 0 0; 0 1 -2; 0 1 2])
  randomize_spins!(sys_min)
  minimize_energy!(sys_min)

  swt = SpinWaveTheory(sys_min;measure = ssf_perp(sys_min))

  q_points = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]];
  qpath = q_space_path(cryst, q_points, 800);
  interact_eigenmodes(swt, qpath)
end
nothing#hide


