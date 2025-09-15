using Sunny, GLMakie, ColorSchemes, LinearAlgebra, JLD2

################################################################################
# Load plotting helpers. Can be eliminated. See note at top of the loaded file.
################################################################################
include("spiralvsskx-plotting.jl")


################################################################################
# Load the simulation results and processed ground states 
################################################################################
data = load("spiralvsskx.jld2")
syss = data["syss"]
res_skx = data["res_skx"]
res_spiral = data["res_spiral"]


################################################################################
# Generate the plot
################################################################################
fig = with_theme(theme_latexfonts()) do
    fig = Figure(; size=(1100, 630))

    # Set up grid layout blocks for whole plot
    g_title_left = fig[1, 1:10]   = GridLayout()
    g_title_right = fig[1, 12:21] = GridLayout()
    g_lt = fig[2:7, 1:10]   =  GridLayout()
    g_rt = fig[2:7, 12:21]  = GridLayout()
    g_lb = fig[8:11, 1:10]  = GridLayout()
    g_rb = fig[8:11, 12:21] = GridLayout()

    # Establish axes for SWT intensities plots
    xticks = res_spiral[1].qpts.xticks
    qs = res_spiral[1].qpts.qs
    axisopts = (;
        ylabel = "Energy (J)",
        titlesize=24,
        xlabel = "Momentum",
        xticks,
        xticklabelsize=20,
        yticklabelsize=20,
        xlabelsize=22,
        ylabelsize=22,
    )
    title0 = L"q_z = 0"
    title1 = L"q_z = 1"
    ax1 = Axis(g_lb[1,1]; title=title0, axisopts...)
    ax2 = Axis(g_lb[1,2]; title=title1, axisopts...)
    ax3 = Axis(g_rb[1,1]; title=title0, axisopts...)
    ax4 = Axis(g_rb[1,2]; title=title1, axisopts...)
    hideydecorations!(ax2)
    hideydecorations!(ax4)

    # Plot intensities into axes
    colormap = reverse(ColorSchemes.roma)
    colorrange1 = (0, 150)
    colorrange2 = (0, 150)
    res = res_spiral[1]
    hm0 = heatmap!(ax1, 1:length(qs), res.energies, res.data'; colorrange=colorrange1, colormap)
    res = res_spiral[2]
    heatmap!(ax2, 1:length(qs), res.energies, res.data'; colorrange=colorrange1, colormap)
    res = res_skx[1]
    hm1 = heatmap!(ax3, 1:length(qs), res.energies, res.data'; colorrange=colorrange2, colormap)
    res = res_skx[2]
    heatmap!(ax4, 1:length(qs), res.energies, res.data'; colorrange=colorrange2, colormap)
    Colorbar(g_lb[1,3], hm0; ticklabelsize=20)
    Colorbar(g_rb[1,3], hm1; ticklabelsize=20)

    # Set up LScenes for spin configuration plots
    lscene1 = LScene(g_lt[1,1]; show_axis=false)
    lscene2 = LScene(g_rt[1,1]; show_axis=false)

    # Plot spin configurations
    sys = syss[1]
    plot_spins_mod!(lscene1, sys; colorfn=i->norm(sys.dipoles[i][3]), colorrange=(0, 0.5), ndims=2, colormap=:roma, show_cell=false, show_bounding_box=false)
    sys = syss[2]
    plot_spins_mod!(lscene2, sys; colorfn=i->norm(sys.dipoles[i][3]), colorrange=(0, 0.5), ndims=2, colormap=:roma, show_cell=false, show_bounding_box=false)
    for lscene in [lscene1, lscene2]
        translate_cam!(lscene.scene, Vec3f(0, -0.5, 0))
        zoom!(lscene.scene, 0.45)
    end

    # Give titles to columns
    Label(g_title_left[1, 1, Top()], "Spiral Phase";
        font = :bold,
        fontsize=26
    )
    Label(g_title_right[1, 1, Top()], "Skyrmion Phase";
        font = :bold,
        fontsize=26
    )
    
    # Add labels that can be referenced in a caption. 
    paddings = [0, 0, 40, 40]
    for (n, (label, layout)) in enumerate(zip(["(a)", "(b)", "(c)", "(d)"], [g_lt, g_rt, g_lb, g_rb]))
        Label(layout[1, 1, TopLeft()], label;
            fontsize=22,
            font = :bold,
            padding = (0, 0, paddings[n], 0),
            halign=:center
        )
    end

    fig   
end
save("../build/spiralvsskx.png", fig)