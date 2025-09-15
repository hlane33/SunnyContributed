using Sunny, LinearAlgebra, JLD2

################################################################################
# Load the saved ground states and metadata
################################################################################
data = load("spiralvsskx-groundstates.jld2")
αs = data["αs"]
Bs = data["Bs"]
coherents_all = data["coherents_all"]


################################################################################
# Functions for setting up the model and processing the data
################################################################################
function tl_dimer_model(; dims=(3, 1, 1), J=1.0, Jp1, Jp2, Jc1, Jc2)
    latvecs = lattice_vectors(1, 1, 10, 90, 90, 120)
    positions = [[0, 0, 0], [0, 0, 0.501]]
    crystal = Crystal(latvecs, positions; types=["B", "T"])

    sys_origin = System(crystal, [1 => Moment(; s=1/2, g=-1), 2 => Moment(; s=1/2, g=-1)], :SUN; dims)

    set_exchange!(sys_origin, J, Bond(1, 2, [0, 0, 0]))
    set_exchange!(sys_origin, Jp1, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys_origin, Jp1, Bond(2, 2, [1, 0, 0]))
    set_exchange!(sys_origin, Jp2, Bond(1, 1, [2, 1, 0]))
    set_exchange!(sys_origin, Jp2, Bond(2, 2, [2, 1, 0]))
    set_exchange!(sys_origin, Jc1, Bond(1, 2, [1, 0, 0]))
    set_exchange!(sys_origin, Jc2, Bond(1, 2, [2, 1, 0]))

    dimers = [(1, 2)]
    sys = Sunny.EntangledSystem(sys_origin, dimers)

    return (; sys, crystal)
end

function tl_dimer_model_from_low_energy_params(; dims=(3, 1, 1), J=1.0, α, Δ) 
    (; Jp1, Jc1, Jp2, Jc2) = Js_from_α_Δ(; J, α, Δ)
    return tl_dimer_model(; dims, J, Jp1, Jp2, Jc1, Jc2) 
end

function Js_from_α_Δ(; J, α, Δ)
    J1 = -α*J
    J2 = 2α*J/(1 + √5)
    Jp1 = (Δ + 0.5)*J1
    Jc1 = (Δ - 0.5)*J1
    Jp2 = (Δ + 0.5)*J2
    Jc2 = (Δ - 0.5)*J2
    return (; Jp1, Jc1, Jp2, Jc2)
end

function one_layer_sys(sys, which=1)
    na, nb, _, _ = size(sys.sys_origin.dipoles)
    sys_single = System(
        Crystal(sys.sys_origin.crystal.latvecs, [[0, 0, 0]]), 
        [1 => Moment(; s=1, g=-1)], 
        :dipole; 
        dims = (na, nb, 1)
    )
    for b in 1:nb, a in 1:na 
        sys_single.dipoles[a, b, 1, which] = sys.sys_origin.dipoles[a, b, 1, which]
    end
    return sys_single
end

function shift_spins_xy!(sys, nx, ny)
    na, nb, _, _ = size(sys.dipoles)
    coherents = copy(sys.coherents)
    dipoles = copy(sys.dipoles)
    for site in eachsite(sys)
        a, b, c, atom = site.I
        sys.coherents[site] = coherents[mod1(a-nx, na), mod1(b-ny, nb), c, atom]
        sys.dipoles[site] = dipoles[mod1(a-nx, na), mod1(b-ny, nb), c, atom]
    end
end


################################################################################
# Perform the calculations
################################################################################
latvecs = lattice_vectors(1, 1, 2, 90, 90, 120)
crystal = Crystal(latvecs, [[0, 0, 0]])
dims = (5, 5, 1)
Δ = 1.2
qs0 = [[0, 0, 0], [2/3, -1/3, 0], [1/2, 0, 0], [0, 0, 0]]
qs1 = [[0, 0, 1], [2/3, -1/3, 1], [1/2, 0, 1], [0, 0, 1]]
path0 = q_space_path(crystal, qs0, 400, labels = ["Γ", "K", "M", "Γ"])
path1 = q_space_path(crystal, qs1, 400, labels = ["Γ", "K", "M", "Γ"])
energies = range(0, 3.0, 400)

syss = []
res_spiral, res_skx = map(zip(αs, Bs, coherents_all)) do (α, B, coherents) 
    (; sys, crystal) = tl_dimer_model_from_low_energy_params(; dims, α, Δ) 
    push!(syss, sys)

    set_field!(sys, [0, 0, B])
    for unit in Sunny.eachunit(sys)
        set_coherent!(sys, coherents[unit], unit)
    end
    minimize_energy!(sys)

    measure = ssf_trace(sys)
    swt = SpinWaveTheory(sys; measure, regularization = 1e-6)

    @time res0 = intensities(swt, path0; energies, kernel=gaussian(; fwhm=0.1))
    @time res1 = intensities(swt, path1; energies, kernel=gaussian(; fwhm=0.1))
    [res0, res1]
end


################################################################################
# Process and save the results 
################################################################################
syss_single_layer = [one_layer_sys(sys) for sys in syss]
shift_spins_xy!(syss_single_layer[1], -1, 0)
shift_spins_xy!(syss_single_layer[2], -1, 0)

# Save the results
syss = syss_single_layer
data = Dict(
    "syss" => syss,
    "res_spiral" => res_spiral,
    "res_skx" => res_skx,
)
save("spiralvsskx.jld2")