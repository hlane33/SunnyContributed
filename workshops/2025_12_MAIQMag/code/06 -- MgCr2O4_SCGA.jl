using Sunny, GLMakie

# Reproduce calculation in Bai et al., Phys. Rev. Lett. 122, 097201 (2019).
latvecs = lattice_vectors(8.3342, 8.3342, 8.3342, 90, 90, 90)
positions = [[1/2, 1/2, 1/2]]
cryst = Crystal(latvecs, positions, 227)
view_crystal(cryst)

sys = System(cryst, [1 => Moment(; s=3/2, g=2)], :dipole)
set_spin_rescaling_for_static_sum_rule!(sys)

# Exchanges from Bai's PRL
J1 = 3.27
set_exchange!(sys, 1.0000*J1, Bond(1, 2, [0, 0, 0]))  # J1
set_exchange!(sys, 0.0815*J1, Bond(1, 7, [0, 0, 0]))  # J2
set_exchange!(sys, 0.1050*J1, Bond(1, 3, [1, 0, 0]))  # J3a
set_exchange!(sys, 0.0085*J1, Bond(1, 3, [0, 0, 0]))  # J3b

formfactors = [1 => FormFactor("Cr3")]
measure = ssf_custom((q, ssf) -> real(ssf[1,1]), sys; apply_g=false, formfactors)
kT = 20*meV_per_K
scga = SCGA(sys; measure, kT, dq=1/4)

fig = Figure(; size=(300, 800))
for (i, l) in enumerate([0, 1/2, 1])
    grid = q_space_grid(cryst, [1, 0, 0], range(0, 4, 100), [0, 1, 0], range(0, 4, 100); offset=[0, 0, l])
    res = Sunny.intensities_static(scga, grid)
    plot_intensities!(fig[i, 1], res)
end
