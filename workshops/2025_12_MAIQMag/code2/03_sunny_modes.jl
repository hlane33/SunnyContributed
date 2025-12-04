using Sunny, GLMakie

latvecs = lattice_vectors(1, 1, 1.2, 90, 90, 90)
cryst = Crystal(latvecs, [[0,0,0]])
view_crystal(cryst)

sys_sun     = System(cryst, (10, 10, 1), [1 => Moment(s=1, g=-1)], :SUN)
sys_rcs     = System(cryst, (10, 10, 1), [1 => Moment(s=1, g=-1)], :dipole)
sys_large_S = System(cryst, (10, 10, 1), [1 => Moment(s=1, g=-1)], :dipole_uncorrected)

begin
    J = 1.0
    h = 0.1 
    D = 0.05

    # Exchange
    set_exchange!(sys_sun, J, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys_rcs, J, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys_large_S, J, Bond(1, 1, [1, 0, 0]))

    # Single-ion anisotropy
    Ss = spin_matrices(1)
    set_onsite_coupling!(sys_sun, D*Ss[3]^2, 1)

    Ss = spin_matrices(1)
    set_onsite_coupling!(sys_rcs, D*Ss[3]^2, 1)

    Ss = spin_matrices(Inf)
    set_onsite_coupling!(sys_large_S, D*Ss[3]^2, 1)

    # External field
    set_external_field!(sys_large_S, h*[0,0,3])
    set_external_field!(sys_rcs, h*[0,0,3])
    set_external_field!(sys_sun, h*[0,0,3])
end


begin
    names = [":dipole_uncorrected (SpinW)", ":SUN", ":dipole"]
    syss = [sys_large_S, sys_sun, sys_rcs]

    fig = Figure(resolution=(1400, 600))
    for (n, (sys, name)) in enumerate(zip(syss, names))
        randomize_spins!(sys)
        minimize_energy!(sys)
        sys_min = reshape_supercell(sys, [1 -1 0; 1 1 0; 0 0 1])
        swt = SpinWaveTheory(sys_min; measure=ssf_trace(sys_min))

        path = q_space_path(cryst, [[0,0,0], [1/2, 1/2,0], [1,1,0]], 500)
        res = intensities_bands(swt, path)

        plot_intensities!(fig[1,n], res; ylims=(0, 8.5), title=name)
    end
    fig
end
