################################################################################
# Code duplication from Sunny with modifications. This could all be eliminated
# by adding a `show_bounding_box` keyword option to plot_spins. In practice, I
# would make a local Sunny branch with the modifications I wanted to avoid this.
################################################################################
function characteristic_length_between_atoms(cryst::Crystal)
    # Detect if atom displacements are on a submanifold (aligned line or plane)
    ps = cryst.positions[1:end-1] .- Ref(cryst.positions[end])
    any_nonzero = map(1:3) do i
        any(p -> !iszero(p[i]), ps)
    end
    vecs = eachcol(cryst.latvecs)[findall(any_nonzero)]

    # Take nth root of appropriate hypervolume per atom
    if length(vecs) == 0
        ℓ = Inf                            # For a single atom, use ℓ0 below
    elseif length(vecs) == 1
        ℓ = norm(vecs[1]) / natoms(cryst)  # Atoms aligned with single lattice vector
    elseif length(vecs) == 2
        ℓ = sqrt(norm(vecs[1] × vecs[2]) / natoms(cryst))
    elseif length(vecs) == 3
        ℓ = cbrt(abs(det(cryst.latvecs)) / natoms(cryst))
    else
        error("Internal error")
    end

    # An upper bound is the norm of the smallest lattice vector.
    ℓ0 = minimum(norm.(eachcol(cryst.latvecs)))

    return min(ℓ0, ℓ)
end

function cell_wireframe(latvecs, ndims)
    vecs = Makie.Point3f.(eachcol(latvecs))
    ret = Tuple{Makie.Point3f, Makie.Point3f}[]

    origin = zero(Makie.Point3f)

    if ndims == 3
        for j in 0:1, k in 0:1
            shift = j*vecs[2]+k*vecs[3]
            push!(ret, (origin+shift, vecs[1]+shift))
        end
        for i in 0:1, k in 0:1
            shift = i*vecs[1]+k*vecs[3]
            push!(ret, (origin+shift, vecs[2]+shift))
        end
        for i in 0:1, j in 0:1
            shift = i*vecs[1]+j*vecs[2]
            push!(ret, (origin+shift, vecs[3]+shift))
        end
    elseif ndims == 2
        for j in 0:1
            shift = j*vecs[2]
            push!(ret, (origin+shift, vecs[1]+shift))
        end
        for i in 0:1
            shift = i*vecs[1]
            push!(ret, (origin+shift, vecs[2]+shift))
        end
    end
    return ret
end

set_alpha(c, alpha) = Makie.coloralpha(c, alpha)

# Analogous to internal Makie function `numbers_to_colors`
function numbers_to_colors!(out::AbstractArray{Makie.RGBAf}, in::AbstractArray{<: Number}, colormap, colorrange)
    @assert length(out) == length(in)
    if isnothing(colorrange) || colorrange[1] >= colorrange[2] - 1e-8
        fill!(out, first(colormap))
    else
        cmin, cmax = colorrange
        len = length(colormap)
        map!(out, in) do c
            # If `cmin ≤ in[i] ≤ cmax` then `0.5 ≤ x ≤ len+0.5`
            x = (c - cmin) / (cmax - cmin) * len + 0.5
            # Round to integer and clip to range [1, len]
            colormap[max(min(round(Int, x), len), 1)]
        end
    end
    return nothing
end

function orient_camera!(ax, latvecs; ghost_radius, ℓ0, orthographic, ndims)
    a1, a2, a3 = eachcol(latvecs)
    if ndims == 3
        lookat = (a1 + a2 + a3)/2
        camshiftdir = normalize(a1 + a2)
        upvector = normalize(a1 × a2)
    elseif ndims == 2
        lookat = (a1 + a2) / 2
        camshiftdir = -normalize(a1 × a2)
        upvector = normalize((a1 × a2) × a1)
    else
        error("Unsupported dimension: $ndims")
    end

    # The extra shift ℓ0 is approximately the nearest-neighbor distance
    camdist = max(cell_diameter(latvecs, ndims)/2 + 0.8ℓ0, ghost_radius)

    orient_camera!(ax; lookat, camshiftdir, upvector, camdist, orthographic)
end

function orient_camera!(ax; lookat, camshiftdir, upvector, camdist, orthographic)
    if orthographic
        eyeposition = lookat - camdist * camshiftdir
        projectiontype = Makie.Orthographic
    else
        eyeposition = lookat - 2.5 * camdist * camshiftdir
        projectiontype = Makie.Perspective
    end

    # Disable the key that would reset camera
    reset = false
    # Do not automatically "recenter" when adding objects
    center = false
    # No rotations on zoom
    zoom_shift_lookat = false
    # Mouse-drag rotations are SO(3) symmetric
    fixed_axis = false

    Makie.cam3d!(ax.scene; lookat, eyeposition, upvector, projectiontype, reset, center, fixed_axis,
                 zoom_shift_lookat, clipping_mode=:view_relative, near=0.01, far=100)
end

function scaled_dipole_to_arrow_geometry(dipole, lengthscale, tiplength)
    # Spin magnitude and direction
    s = norm(dipole)
    dir = dipole / s

    # In the typical case, spin magnitude will be denoted by shaft length.
    shaftlength0 = s * lengthscale

    # If spin magnitude is too small, reduce overall arrow length by the factor
    # c ~ cbrt(s) ≤ 1. Here, the spin magnitude is effectively represented by
    # the _volume_ of the arrow tip.
    r = shaftlength0 / (0.5 * tiplength)
    c = cbrt(min(r, 1))
    full_length = c * (shaftlength0 + tiplength)

    # The true space remaining for the shaft. If no space is left, Makie will
    # also rescale the arrow tip as needed to achieve the requested full_length.
    shaftlength = max(full_length - tiplength, 0)

    offset = -(shaftlength/2) * dir
    vec = full_length * dir
    return (; offset, vec)
end

function cell_diameter(latvecs, ndims)
    (a1, a2, a3) = eachcol(latvecs)
    if ndims == 3
        return max(norm(a1+a2+a3), norm(a1+a2-a3), norm(a1-a2+a3), norm(a1-a2-a3))
    elseif ndims == 2
        return max(norm(a1+a2), norm(a1-a2))
    else
        error("Unsupported `ndims=$ndims`.")
    end
end

function cell_center(ndims)
    if ndims == 3
        return [1, 1, 1] / 2
    elseif ndims == 2
        return [1, 1, 0] / 2
    else
        error("Unsupported `ndims=$ndims`.")
    end
end

function plot_spins_mod(sys::System; size=(768, 512), compass=true, kwargs...)
    fig = Makie.Figure(; size)
    ax = Makie.LScene(fig[1, 1]; show_axis=false)
    notifier = Makie.Observable(nothing)
    plot_spins_mod!(ax, sys; notifier, kwargs...)
    compass && add_cartesian_compass(fig, ax)
    return NotifiableFigure(notifier, fig)
end

function plot_spins_mod!(ax, sys::Sunny.System; notifier=Makie.Observable(nothing), arrowscale=1.0, stemcolor=:lightgray, color=:red,
                     colorfn=nothing, colormap=:viridis, colorrange=nothing, show_cell=true, orthographic=false,
                     ghost_radius=0, ndims=3, dims=nothing, show_bounding_box=true)
    isnothing(dims) || error("Use notation `ndims=$dims` instead of `dims=$dims`")

    if ndims == 2
        sys.dims[3] == 1 || error("System not two-dimensional in (a₁, a₂)")
    elseif ndims == 1
        sys.dims[[2,3]] == [1,1] || error("System not one-dimensional in (a₁)")
    end

    # Show bounding box of magnetic supercell in gray (this needs to come first
    # to set a scale for the scene in case there is only one atom).
    supervecs = sys.crystal.latvecs * diagm(Sunny.Vec3(sys.dims))
    if show_bounding_box
        # Makie.linesegments!(ax, cell_wireframe(supervecs, ndims); color=:gray, linewidth=1.5)
    end

    # Infer characteristic length scale between sites
    ℓ0 = characteristic_length_between_atoms(Sunny.orig_crystal(sys))

    # Quantum spin-s averaged over sites. Will be used to normalize dipoles.
    N0 = norm(sys.Ns) / sqrt(length(sys.Ns))
    s0 = (N0 - 1) / 2

    # Parameters defining arrow shape
    a0 = arrowscale * ℓ0
    markersize = 0.1*a0
    shaftradius = 0.06a0
    tipradius = 0.2a0
    tiplength = 0.4a0
    lengthscale = 0.6a0

    # Positions in fractional coordinates of supercell vectors
    rs = [supervecs \ global_position(sys, site) for site in eachsite(sys)]

    for isghost in (false, true)
        if isghost
            alpha = 0.08
            (idxs, offsets) = Sunny.all_offsets_within_distance(supervecs, rs, cell_center(ndims); max_dist=ghost_radius, nonzeropart=true)
        else
            alpha = 1.0
            idxs = eachindex(rs)
            offsets = [zero(Vec3) for _ in idxs]
        end

        # Every call to RGBf constructor allocates, so pre-calculate color
        # arrays to speed animations
        cmap_with_alpha = set_alpha.(Makie.to_colormap(colormap), Ref(alpha))
        numeric_colors = zeros(size(sys.dipoles))
        rgba_colors = zeros(Makie.RGBAf, size(sys.dipoles))

        if isnothing(colorfn)
            # In this case, we can precompute the fixed `rgba_colors` array
            # according to `color`
            if color isa AbstractArray
                @assert length(color) == length(sys.dipoles)
                if eltype(color) <: Number
                    dyncolorrange = @something colorrange extrema(color)
                    numbers_to_colors!(rgba_colors, color, cmap_with_alpha, dyncolorrange)
                else
                    map!(rgba_colors, color) do c
                        set_alpha(Makie.to_color(c), alpha)
                    end
                end
            else
                c = set_alpha(Makie.to_color(color), alpha)
                fill!(rgba_colors, c)
            end
        end

        # These observables will be reanimated upon calling `notify(notifier)`.
        pts = Makie.Observable(Makie.Point3f[])
        offset_pts = Makie.Observable(Makie.Point3f[])
        vecs = Makie.Observable(Makie.Vec3f[])
        tipcolor = Makie.Observable(Makie.RGBAf[])

        Makie.on(notifier, update=true) do _
            empty!.((vecs[], offset_pts[], tipcolor[]))

            # Dynamically adapt `rgba_colors` according to `colorfn`
            if !isnothing(colorfn)
                numeric_colors .= colorfn.(CartesianIndices(sys.dipoles))
                dyncolorrange = @something colorrange extrema(numeric_colors)
                numbers_to_colors!(rgba_colors, numeric_colors, cmap_with_alpha, dyncolorrange)
            end

            for (site, n) in zip(idxs, offsets)
                (; offset, vec) = scaled_dipole_to_arrow_geometry(sys.dipoles[site]/s0, lengthscale, tiplength)
                pt = supervecs * (rs[site] + n)
                push!(pts[], Makie.Point3f(pt))
                push!(offset_pts[], pt + Makie.Point3f(offset))
                push!(vecs[], Makie.Vec3f(vec))
                push!(tipcolor[], rgba_colors[site])
            end
            # Trigger Makie redraw
            notify.((offset_pts, vecs, tipcolor))
            # isnothing(color) || notify(arrowcolor)
        end

        # Draw arrows
        shaftcolor = (stemcolor, alpha)
        if !isempty(offset_pts[])
            Makie.arrows3d!(ax, offset_pts, vecs; align=0, markerscale=1, minshaftlength=0, tipradius,
                            shaftradius, tiplength, tipcolor, shaftcolor, diffuse=1.15, transparency=isghost)
        end

        # Small sphere inside arrow to mark atom position
        Makie.meshscatter!(ax, pts; markersize, color=shaftcolor, diffuse=1.15, transparency=isghost)
    end

    # Bounding box of original crystal unit cell in teal
    if show_cell
        Makie.linesegments!(ax, cell_wireframe(Sunny.orig_crystal(sys).latvecs, ndims); color=:teal, linewidth=1.5)
        pos = [(3/4)*Makie.Point3f(p) for p in eachcol(Sunny.orig_crystal(sys).latvecs)[1:ndims]]
        text = [Makie.rich("a", Makie.subscript(repr(i))) for i in 1:ndims]
        Makie.text!(ax, pos; text, color=:black, fontsize=20, font=:bold, glowwidth=4.0,
                    glowcolor=(:white, 0.6), align=(:center, :center), depth_shift=-1f0)
    end

    orient_camera!(ax, supervecs; ghost_radius, ℓ0, orthographic, ndims)

    return ax
end
