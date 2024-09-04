using Sunny, StaticArrays, Printf
"""
    BinningParameters(binstart,binend,binwidth;covectors = I(4))
    BinningParameters(binstart,binend;numbins,covectors = I(4))

Describes a 4D parallelepided histogram in a format compatible with experimental
Inelasitic Neutron Scattering data. See
[`generate_mantid_script_from_binning_parameters`](@ref) to convert
[`BinningParameters`](@ref) to a format understandable by the [Mantid
software](https://www.mantidproject.org/), or [`load_nxs`](@ref) to load
[`BinningParameters`](@ref) from a Mantid `.nxs` file.
 
The coordinates of the histogram axes are specified by multiplication of `(q,ω)`
with each row of the `covectors` matrix, with `q` given in [R.L.U.]. Since the
default `covectors` matrix is the identity matrix, the default axes are
`(qx,qy,qz,ω)` in absolute units.

The convention for the binning scheme is that:
- The left edge of the first bin starts at `binstart`
- The bin width is `binwidth`
- The last bin contains `binend`
- There are no "partial bins;" the last bin may contain values greater than
  `binend`.
  
A `value` can be binned by computing its bin index:

    coords = covectors * value
    bin_ix = 1 .+ floor.(Int64,(coords .- binstart) ./ binwidth)
"""
mutable struct BinningParameters
    binstart::MVector{4,Float64}
    binend::MVector{4,Float64}
    binwidth::MVector{4,Float64}
    covectors::MMatrix{4,4,Float64}
end
# TODO: Use the more efficient three-argument `div(a,b,RoundDown)` instead of `floor(a/b)`
# to implement binning. Both performance and correctness need to be checked.

function Base.show(io::IO, ::MIME"text/plain", params::BinningParameters)
    printstyled(io, "Binning Parameters\n"; bold=true, color=:underline)
    nbin = params.numbins
    for k = 1:4
        if nbin[k] == 1
            printstyled(io, "∫ Integrated"; bold=true)
        else
            printstyled(io, @sprintf("⊡ %5d bins",nbin[k]); bold=true)
        end
        bin_edges = axes_binedges(params)
        first_edges = map(x -> x[1],bin_edges)
        last_edges = map(x -> x[end],bin_edges)
        @printf(io," from %+.3f to %+.3f along [", first_edges[k], last_edges[k])
        print(io,covector_name(params.covectors[k,:]))
        @printf(io,"] (Δ = %.3f)", params.binwidth[k]/norm(params.covectors[k,:]))
        println(io)
    end
end

function covector_name(cov)
  str = ""
  axes_names = ["x","y","z","E"]
  inMiddle = false
  for j = 1:4
      if cov[j] != 0.
          if(inMiddle)
              str *= " "
          end
          str *= @sprintf("%+.2f d%s",cov[j],axes_names[j])
          inMiddle = true
      end
  end
  str
end

# Creates a binning scheme centered on the q_path, with the specified transverse
# binning directions and bin widths.
function specify_transverse_binning(q_path::Sunny.QPath, first_transverse_axis, second_transverse_axis, first_transverse_binwidth, second_transverse_binwidth)
  # Ensure path is non-empty and single-segment
  if length(q_path.qs) < 2
    error("Q path must have at least two points to infer bin width")
  end

  Δq = q_path.qs[2] - q_path.qs[1]

  if !all([Δq ≈ dq for dq = diff(q_path.qs)])
    error("Q path is multi-segment or irregular!")
  end

  binstart = zero(MVector{4,Float64})
  binstart[4] = -Inf # Default to integrate over all energies

  binend = zero(MVector{4,Float64})
  binend[4] = 0

  covectors = zero(MMatrix{4,4,Float64})
  recip_directions = zeros(Float64,3,3)
  recip_directions[:,1] .= Δq ./ norm(Δq)
  recip_directions[:,2] .= first_transverse_axis
  recip_directions[:,3] .= second_transverse_axis

  if minimum(svd(recip_directions).S) < 1e-8
    error("Axes are collinear!")
  end
  covectors[1:3,1:3] .= inv(recip_directions)

  coords_start = covectors[1:3,1:3] * q_path.qs[1]
  coords_end = covectors[1:3,1:3] * q_path.qs[end]

  binwidth = zero(MVector{4,Float64})
  binwidth[1] = (covectors[1:3,1:3] * Δq)[1]
  binwidth[2] = first_transverse_binwidth
  binwidth[3] = second_transverse_binwidth
  binwidth[4] = Inf

  binstart[1:3] .= coords_start[1:3] .- binwidth[1:3]/2
  binend[1:3] .= coords_end[1:3]

  # Check the original Q points end up in bin centers
  in_bin(q) = (covectors[1:3,1:3] * q .- binstart[1:3]) ./ binwidth[1:3]
  centering_error(q) = (in_bin(q) .- 0.5) .- (round.(Int64,in_bin(q) .- 0.5))
  @assert all([norm(centering_error(q_path.qs[i])) < 1e-12 for i = 1:length(q_path.qs)])

  # Energy axis
  covectors[4,:] .= [0,0,0,1]

  BinningParameters(binstart,binend,binwidth,covectors)
end

# Creates a binning scheme centered on the q_grid, with the specified transverse
# binning direction and bin width.
function specify_transverse_binning(q_grid::Sunny.QGrid{2}, transverse_axis, transverse_binwidth)
  # Ensure grid is non-empty and single-segment
  if size(q_grid.qs,1) < 2 || size(q_grid.qs,2) < 2
    error("2D Q grid must have at least two points in each direction")
  end

  Δq1 = q_grid.qs[2,1] - q_grid.qs[1,1]
  Δq2 = q_grid.qs[1,2] - q_grid.qs[1,1]

  if !all([Δq1 ≈ dq for dq = diff(q_grid.qs,dims=1)]) || !all([Δq2 ≈ dq for dq = diff(q_grid.qs,dims = 2)])
    error("2D Q grid is irregular!")
  end

  binstart = zero(MVector{4,Float64})
  binstart[4] = -Inf # Default to integrate over all energies

  binend = zero(MVector{4,Float64})
  binend[4] = 0

  covectors = zero(MMatrix{4,4,Float64})
  recip_directions = zeros(Float64,3,3)
  recip_directions[:,1] .= Δq1 ./ norm(Δq1)
  recip_directions[:,2] .= Δq2 ./ norm(Δq2)
  recip_directions[:,3] .= transverse_axis

  if minimum(svd(recip_directions).S) < 1e-8
    error("Transverse axis is in-plane!")
  end
  covectors[1:3,1:3] .= inv(recip_directions)

  coords_start = covectors[1:3,1:3] * q_grid.qs[1]
  coords_end = covectors[1:3,1:3] * q_grid.qs[end]

  first_binwidth = (covectors[1:3,1:3] * Δq1)[1]
  second_binwidth = (covectors[1:3,1:3] * Δq2)[2]

  binwidth = zero(MVector{4,Float64})
  binwidth[1] = first_binwidth
  binwidth[2] = second_binwidth
  binwidth[3] = transverse_binwidth
  binwidth[4] = Inf

  binstart[1:3] .= coords_start[1:3] .- binwidth[1:3]/2
  binend[1:3] .= coords_end[1:3]

  # Check the original Q points end up in bin centers
  in_bin(q) = (covectors[1:3,1:3] * q .- binstart[1:3]) ./ binwidth[1:3]
  centering_error(q) = (in_bin(q) .- 0.5) .- (round.(Int64,in_bin(q) .- 0.5))
  @assert all([norm(centering_error(q_grid.qs[i])) < 1e-8 for i = 1:length(q_grid.qs)])

  # Energy axis
  covectors[4,:] .= [0,0,0,1]
  BinningParameters(binstart,binend,binwidth,covectors)
end

"""
    unit_resolution_binning_parameters(sc::SampledCorrelations)

Create [`BinningParameters`](@ref) which place one histogram bin centered at each possible `(q,ω)` scattering vector of the crystal.
This is the finest possible binning without creating bins with zero scattering vectors in them.
"""
function unit_resolution_binning_parameters(sc::SampledCorrelations;negative_energies = true)
    ωvals = available_energies_including_zero(sc;negative_energies)

    good_qs = Sunny.available_wave_vectors(sc)
    numbins = (size(good_qs)...,length(ωvals))
    # Bin centers should be at Sunny scattering vectors
    maxQ = 1 .- (1 ./ numbins)
    
    min_val = (0.,0.,0.,minimum(ωvals))
    max_val = (maxQ[1],maxQ[2],maxQ[3],maximum(ωvals))
    total_size = max_val .- min_val

    binwidth = total_size ./ (numbins .- 1)
    binstart = (0.,0.,0.,minimum(ωvals)) .- (binwidth ./ 2)
    binend = (maxQ[1],maxQ[2],maxQ[3],maximum(ωvals)) # bin end is well inside of last bin

    params = BinningParameters(binstart,binend,binwidth,I(4))

    # Special case for when there is only one bin in a direction
    for i = 1:4
        if numbins[i] == 1
            params.binwidth[i] = 1.
            params.binstart[i] = min_val[i] - (params.binwidth[i] ./ 2)
            params.binend[i] = min_val[i]
        end
    end
    params
end

unit_resolution_binning_parameters(sc::SampledCorrelationsStatic;kwargs...) = unit_resolution_binning_parameters(sc.parent;kwargs...)

Base.copy(p::BinningParameters) = BinningParameters(copy(p.binstart),copy(p.binend),copy(p.binwidth),copy(p.covectors))

# Support numbins as a (virtual) property, even though only the binwidth is stored
Base.getproperty(params::BinningParameters, sym::Symbol) = sym == :numbins ? [count_bins(params.binstart[i],params.binend[i],params.binwidth[i]) for i = 1:4] : getfield(params,sym)

function Base.setproperty!(params::BinningParameters, sym::Symbol, numbins)
    if sym == :numbins
        # *Ensure* that the last bin contains params.binend
        params.binwidth .= (params.binend .- params.binstart) ./ (numbins .- 0.5)
    else
        setfield!(params,sym,numbins)
    end
end

"""
    count_bins(binstart,binend,binwidth)

Returns the number of bins in the binning scheme implied by `binstart`, `binend`, and `binwidth`.
To count the bins in a [`BinningParameters`](@ref), use `params.numbins`.

This function defines how partial bins are handled, so it should be used preferentially over
computing the number of bins manually.
"""
function count_bins(bin_start,bin_end,bin_width)
  if !isfinite(bin_width)
    1
  else
    ceil(Int64,(bin_end - bin_start) / bin_width)
  end
end

"""
    axes_bincenters(params::BinningParameters)

Returns tick marks which label the bins of the histogram described by [`BinningParameters`](@ref) by their bin centers.

The following alternative syntax can be used to compute bin centers for a single axis:

    axes_bincenters(binstart,binend,binwidth)
"""
function axes_bincenters(binstart,binend,binwidth)
    bincenters = Vector{AbstractVector{Float64}}(undef,0)
    for k = eachindex(binstart)
        if isfinite(binwidth[k])
          first_center = binstart[k] .+ binwidth[k] ./ 2
          nbin = count_bins(binstart[k],binend[k],binwidth[k])
          push!(bincenters,range(first_center,step = binwidth[k],length = nbin))
        else
          push!(bincenters,[binstart[k]])
        end
    end
    bincenters
end
axes_bincenters(params::BinningParameters) = axes_bincenters(params.binstart,params.binend,params.binwidth)

function axes_binedges(binstart,binend,binwidth)
    binedges = Vector{AbstractVector{Float64}}(undef,0)
    for k = eachindex(binstart)
        if isfinite(binwidth[k])
          nbin = count_bins(binstart[k],binend[k],binwidth[k])
          push!(binedges,range(binstart[k],step = binwidth[k],length = nbin + 1))
        else
          push!(binedges,[-Inf,Inf])
        end
    end
    binedges
end
axes_binedges(params::BinningParameters) = axes_binedges(params.binstart,params.binend,params.binwidth)

# Find an axis-aligned bounding box containing the histogram
function binning_parameters_aabb(params)
    (; binstart, binend, covectors) = params
    bin_edges = axes_binedges(params)
    first_edges = map(x -> x[1],bin_edges)
    last_edges = map(x -> x[end],bin_edges)
    bin_edges = [first_edges last_edges]
    this_corner = MVector{4,Float64}(undef)
    q_corners = MMatrix{4,16,Float64}(undef)
    for j = 1:16 # The sixteen corners of a 4-cube
        for k = 1:4 # The four axes
            this_corner[k] = bin_edges[k,1 + (j >> (k-1) & 1)]
        end
        this_corner[.!isfinite.(this_corner)] .= 0
        q_corners[:,j] = covectors \ this_corner
    end
    lower_aabb_q = minimum(q_corners,dims=2)[1:3]
    upper_aabb_q = maximum(q_corners,dims=2)[1:3]
    return lower_aabb_q, upper_aabb_q
end

struct BinnedIntensities{T} <: Sunny.AbstractIntensities
    # Original chemical cell
    #crystal :: Crystal
    # BinningParameters in RLU
    params :: BinningParameters
    # Intensity data as bin-integrated values
    data :: Array{T, 4} # collect(size(data)) == params.numbins
    # Number of individually binned contributions (useful for some normalizations)
    counts :: Array{Float64, 4}
end

function Base.show(io::IO, res::BinnedIntensities)
    sz = join(res.params.numbins, "×")
    print(io, string(typeof(res)) * " ($sz bins)")
end

function binned_intensities(sc,params::BinningParameters;kT = nothing,integrated_kernel = nothing)
    static_mode = sc isa SampledCorrelationsStatic
    if !isnothing(integrated_kernel) && static_mode
      error("Can't broaden if data is not energy-resolved")
    end

    # Decide on which Q points can possibly contribute (depends on geometry of
    # supercell and params)
    lower_aabb_q, upper_aabb_q = binning_parameters_aabb(params)
    # Round the axis-aligned bounding box *outwards* to lattice sites
    # SQTODO: are these bounds optimal?
    Ls = size((static_mode ? sc.parent : sc).data)[4:6]
    lower_aabb_cell = floor.(Int64,lower_aabb_q .* Ls .+ 1) 
    upper_aabb_cell = ceil.(Int64,upper_aabb_q .* Ls .+ 1) 
    cells = CartesianIndices(Tuple(((:).(lower_aabb_cell,upper_aabb_cell))))[:]
    qpts = Sunny.QPoints([Vec3((cell.I .- 1) ./ Ls) for cell = cells])

    # Calculate intensity at prepared qpts. No broadening yet because
    # that depends on the bin layout and uses an integrated_kernel!
    energies = if static_mode
            [0.0]
        else
            sort(available_energies_including_zero(sc;negative_energies=true))
        end
    res = if static_mode
      intensities_static(sc, qpts)
    else
      intensities(sc, qpts; energies, kT)
    end

    # Bin (and broaden) those intensities according to BinningParameters
    k = MVector{3,Float64}(undef)
    v = MVector{4,Float64}(undef)
    q = view(v,1:3)
    coords = MVector{4,Float64}(undef)
    xyztBin = MVector{4,Int64}(undef)
    xyzBin = view(xyztBin,1:3)

    (; binwidth, binstart, binend, covectors, numbins) = params
    return_type = typeof(res).parameters[1]
    output_intensities = zeros(return_type,numbins...)
    output_counts = zeros(Float64,numbins...)

    # Pre-compute discrete broadening kernel from continuous one provided
    if !isnothing(integrated_kernel)
        # Upgrade to 2-argument kernel if needed
        integrated_kernel_edep = try
            integrated_kernel(0.,0.)
            integrated_kernel
        catch MethodError
            (ω,Δω) -> integrated_kernel(Δω)
        end

        fraction_in_bin = Vector{Vector{Float64}}(undef,length(energies))
        for (iω,ω) in enumerate(energies)
            fraction_in_bin[iω] = Vector{Float64}(undef,numbins[4])
            for iωother = 1:numbins[4]
                ci_other = CartesianIndex(xyzBin[1],xyzBin[2],xyzBin[3],iωother)
                # Start and end points of the target bin
                a = binstart[4] + (iωother - 1) * binwidth[4]
                b = binstart[4] + iωother * binwidth[4]

                # P(ω picked up in bin [a,b]) = ∫ₐᵇ Kernel(ω' - ω) dω'
                fraction_in_bin[iω][iωother] = integrated_kernel_edep(ω,b - ω) - integrated_kernel_edep(ω,a - ω)
            end
        end
    end

    for cell_ix = 1:length(cells), (iω,ω) in enumerate(energies)
        cell = cells[cell_ix]
        q .= ((cell.I .- 1) ./ Ls) # q is in R.L.U.
        k .= (static_mode ? sc.parent : sc).crystal.recipvecs * q
        if isnothing(integrated_kernel) # `Delta-function energy' logic
            # Figure out which bin this goes in
            v[4] = ω
            mul!(coords,covectors,v)
            coords .= (coords .- binstart) ./ binwidth
            coords[.!isfinite.(binwidth)] .= 0
            xyztBin .= 1 .+ floor.(Int64,coords)

            # Check this bin is within the 4D histogram bounds
            if all(xyztBin .<= numbins) && all(xyztBin .>= 1)
                intensity = static_mode ? res.data[cell_ix] : res.data[iω,cell_ix]

                ci = CartesianIndex(xyztBin.data)
                output_intensities[ci] += intensity
                output_counts[ci] += 1
            end
        else # `Energy broadening into bins' logic
            # For now, only support broadening for `simple' energy axes
            if covectors[4,:] == [0,0,0,1] && norm(covectors[1:3,:] * [0,0,0,1]) == 0

                # Check this bin is within the *spatial* 3D histogram bounds
                # If we are energy-broadening, then scattering vectors outside the histogram
                # in the energy direction need to be considered
                mul!(view(coords,1:3),view(covectors,1:3,1:3), view(v,1:3))
                xyzBin .= 1 .+ floor.(Int64,(view(coords,1:3) .- view(binstart,1:3)) ./ view(binwidth,1:3))
                if all(xyzBin .<= view(numbins,1:3)) &&  all(xyzBin .>= 1)

                    # Calculate source scattering vector intensity only once
                    intensity = res.data[iω,cell_ix]

                    # Broaden from the source scattering vector (k,ω) to
                    # each target bin ci_other
                    ci_other = CartesianIndex(xyzBin[1],xyzBin[2],xyzBin[3])
                    view(output_intensities,ci_other,:) .+= fraction_in_bin[iω] .* Ref(intensity)
                    view(output_counts,ci_other,:) .+= fraction_in_bin[iω]
                end
            else
                error("Energy broadening not yet implemented for histograms with complicated energy axes")
            end
        end
    end
    N_bins_in_BZ = abs(det(covectors[1:3,1:3])) / prod(binwidth[1:3])
    output_data = output_intensities ./ N_bins_in_BZ ./ length(energies)
    BinnedIntensities(params,output_data,output_counts)
end

"""
    integrate_axes!(params::BinningParameters; axes)
Integrate over one or more axes of the histogram by setting the number of bins
in that axis to 1. Examples:

    integrate_axes!(params; axes = [2,3])
    integrate_axes!(params; axes = 2)
"""
function integrate_axes!(params::BinningParameters;axes)
    for k in axes
        nbins = [params.numbins.data...]
        nbins[k] = 1
        params.numbins = SVector{4}(nbins)
    end
    return params
end

function energy_resolve!(params::BinningParameters,energies)
  energies = sort(energies)
  params.binend[4] = maximum(energies)
  params.binwidth[4] = energies[2] - energies[1]
  params.binstart[4] = energies[1] - params.binwidth[4]/2
  params
end

function available_energies_including_zero(x; kwargs...)
    ωs = Sunny.available_energies(x;kwargs...)
    # Special case due to NaN definition of instant_correlations
    (length(ωs) == 1 && isnan(ωs[1])) ? [0.] : ωs
end

"""
    generate_mantid_script_from_binning_parameters(params::BinningParameters)

Generate a Mantid script which bins data according to the given
[`BinningParameters`](@ref).

!!! warning "Units"

    Take care to ensure the units are correct (R.L.U. or absolute).
    You may want to call `Sunny.bin_rlu_as_absolute_units!` or
    `Sunny.bin_absolute_units_as_rlu!` first.
"""
function generate_mantid_script_from_binning_parameters(params)
    covectorsK = params.covectors # Please call rlu_to_absolute_units! first if needed
    #function bin_string(k)
        #if params.numbins[k] == 1
            #return "$(params.binsstart[k]),$(params.binend[k])"
        #else
            #return "$(params.binsstart[k]),$(params.binend[k])"
        #end
    #end
    return """MakeSlice(InputWorkspace="merged_mde_INPUT",
        QDimension0="$(covectorsK[1,1]),$(covectorsK[1,2]),$(covectorsK[1,3])",
        QDimension1="$(covectorsK[2,1]),$(covectorsK[2,2]),$(covectorsK[2,3])",
        QDimension2="$(covectorsK[3,1]),$(covectorsK[3,2]),$(covectorsK[3,3])",
        Dimension0Binning="$(params.binstart[1]),$(params.binwidth[1]),$(params.binend[1])",
        Dimension1Name="DeltaE",
        Dimension1Binning="$(params.binstart[2]),$(params.binwidth[2]),$(params.binend[2])",
        Dimension2Binning="$(params.binstart[3]),$(params.binwidth[3]),$(params.binend[3])",
        Dimension3Binning="$(params.binstart[4]),$(params.binwidth[4]),$(params.binend[4])",
        Dimension3Name="QDimension1",
        Smoothing="0",
        OutputWorkspace="Histogram_OUTPUT")
        """
end

using JLD2
"""
    params, signal = load_nxs(filename; field="signal")

Given the name of a Mantid-exported `MDHistoWorkspace` file, load the
[`BinningParameters`](@ref) and the signal from that file.

To load another field instead of the signal, specify e.g.
`field="errors_squared"`. Typical fields include `errors_squared`, `mask`,
`num_events`, and `signal`.
"""
function load_nxs(filename; field="signal")
    JLD2.jldopen(filename,"r") do file
        read_covectors_from_axes_labels = false

        # This variable is basically the "Mantid W-Matrix". See discussion on
        # Github: https://github.com/SunnySuite/Sunny.jl/pull/256.
        spatial_covectors = Matrix{Float64}(undef,3,3)
        try
          try
            w_matrix = file["MDHistoWorkspace"]["experiment0"]["logs"]["W_MATRIX"]["value"]

            # Transpose to arrange axes labels as columns
            spatial_covectors .= transpose(reshape(w_matrix,3,3))
          catch e
            printstyled("Warning",color=:yellow)
            print(": failed to load W_MATRIX from Mantid file $filename due to:\n")
            println(e)
            println("Falling back to reading transform_from_orig")

            coordinate_system = file["MDHistoWorkspace"]["coordinate_system"][1]

            # Permalink to where this enum is defined:
            # https://github.com/mantidproject/mantid/blob/057df5b2de1c15b819c7dd06e50bdbf5461b09c6/Framework/Kernel/inc/MantidKernel/SpecialCoordinateSystem.h#L14
            system_name = ["None", "QLab", "QSample", "HKL"][coordinate_system + 1]

            # The matrix stored in the file is transpose of the actual
            # transform_from_orig matrix
            transform_from_orig = file["MDHistoWorkspace"]["transform_from_orig"]
            transform_from_orig = reshape(transform_from_orig,5,5)
            transform_from_orig = transform_from_orig'
            
            # U: Orthogonal rotation matrix
            # B: inv(lattice_vectors(...)) matrix, as in Sunny
            # The matrix stored in the file is transpose of U * B
            ub_matrix = file["MDHistoWorkspace"]["experiment0"]["sample"]["oriented_lattice"]["orientation_matrix"]
            ub_matrix = ub_matrix'

            # Following: https://docs.mantidproject.org/nightly/concepts/Lattice.html
            # It can be verified that the matrix G^* = (ub_matrix' * ub_matrix)
            # is equal to B' * B, where B = inv(lattice_vectors(...)), and the diagonal
            # entries of the inverse of this matrix are the lattice parameters squared
            #
            #abcMagnitude = sqrt.(diag(inv(ub_matrix' * ub_matrix)))
            #println("[a,b,c] = ",abcMagnitude)

            # This is how you extract the covectors from `transform_from_orig` and `ub_matrix`
            # TODO: Parse this from the `long_name` of the data_dims instead
            spatial_covectors .= 2π .* transform_from_orig[1:3,1:3] * ub_matrix
          end
        catch e
          printstyled("Warning",color=:yellow)
          print(": failed to read histogram axes from Mantid file $filename due to:\n")
          println(e)
          println("Defaulting to low-accuracy reading of axes labels!")
          read_covectors_from_axes_labels = true
        end

        signal = file["MDHistoWorkspace"]["data"][field]

        axes = Dict(JLD2.load_attributes(file,"MDHistoWorkspace/data/signal"))[:axes]

        # Axes are just stored backwards in Mantid .nxs for some reason
        axes_names = reverse(split(axes,":"))

        data_dims = Vector{Vector{Float64}}(undef,length(axes_names))
        binwidth = Vector{Float64}(undef,4)
        binstart = Vector{Float64}(undef,4)
        binend = Vector{Float64}(undef,4)
        covectors = zeros(4, 4)
        spatial_covector_ixs = [0,0,0]
        std = x -> sqrt(sum((x .- sum(x) ./ length(x)).^2))
        for (i,name) in enumerate(axes_names)
            long_name = Dict(JLD2.load_attributes(file,"MDHistoWorkspace/data/$name"))[:long_name]

            if long_name == "DeltaE"
                covectors[i,:] .= [0,0,0,1] # energy covector
            else # spatial covector case
                ix = findfirst(spatial_covector_ixs .== 0)
                spatial_covector_ixs[ix] = i
                if read_covectors_from_axes_labels
                    lbl = parse_long_name(long_name)
                    spatial_covectors[:,ix] .= lbl
                end
            end

            data_dims[i] = file["MDHistoWorkspace"]["data"][name]
            binwidth[i] = sum(diff(data_dims[i])) / length(diff(data_dims[i]))
            if std(diff(data_dims[i])) > 1e-4 * binwidth[i]
              printstyled("Warning possible non-uniform binning: mean width = $(binwidth[i]),  std width = $(std(diff(data_dims[i])))"; color = :yellow)
            end

            binstart[i] = minimum(data_dims[i])

            # Place end of bin in center of last bin, according to Sunny convention
            binend[i] = maximum(data_dims[i]) - binwidth[i]/2
        end

        covectors[spatial_covector_ixs,1:3] .= inv(spatial_covectors)

        return BinningParameters(binstart,binend,binwidth,covectors), signal
    end
end

function Base.permutedims(params::BinningParameters,perm)
  out = copy(params)
  out.covectors .= params.covectors[perm,:]
  out.binwidth .= params.binwidth[perm]
  out.binstart .= params.binstart[perm]
  out.binend .= params.binend[perm]
  out
end

# Parse the "[0.5H,0.3H,0.1H]" type of Mantid string describing
# a histogram axis
function parse_long_name(long_name)
  # You're welcome
  found = match(r"\[(|0|(?:-?[0-9]?(?:\.[0-9]+)?))[HKL]?,(|0|(?:-?[0-9]?(?:\.[0-9]+)?))[HKL]?,(|0|(?:-?[0-9]?(?:\.[0-9]+)?))[HKL]?\]",long_name)
  if isnothing(found)
    return nothing
  end
  return map(x -> isempty(x) ? 1. : x == "-" ? -1. : parse(Float64,x),found)
end

function quick_view_nxs(filename,keep_ax)
    integration_axes = setdiff(1:4,keep_ax)
    params, signal = load_nxs(filename)
    integrate_axes!(params, axes=integration_axes)
    int_signal = dropdims(sum(signal, dims=integration_axes); dims=Tuple(integration_axes))
    bcs = axes_bincenters(params)
    (bcs[keep_ax[1]], bcs[keep_ax[2]], int_signal)
end