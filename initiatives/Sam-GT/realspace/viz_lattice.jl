using Sunny, GLMakie, LinearAlgebra, Observables

include("higher_swt.jl")
include("viz_hist.jl")

function view_lattice(cryst;params = nothing,latsize = (1,1,1))
  if !isnothing(params)
    @assert iszero(params.covectors[1:3,4]) && iszero(params.covectors[4,1:3])
  end
  na = length(cryst.positions)
  rs = abs(det(cryst.latvecs))^(1/3)

  f = Figure()
  controls = GridLayout(f[3,1],tellwidth = false)
  txt_tog = Toggle(controls[1,1])
  Label(controls[1,2];text = "Toggle labels")
  Label(controls[2,1];text = "rₛ=$(Sunny.number_to_simple_string(rs;digits = 4))Å")

  ax_real_lattice = LScene(f[1,1];show_axis = false)#, title = "Lattice (Direct)")

  lat_points = Point3f[]
  lat_points_rlu = Point3f[]
  point_sites = Int64[]
  nbzs = polyatomic_bzs(cryst)

  recip_lat_comm_points = Point3f[]
  recip_lat_abs_comm_points = Point3f[]

  for lat_loc = CartesianIndices(latsize)
    for i = 1:na
      push!(lat_points,Point3f(cryst.latvecs * (collect(lat_loc.I .- 1) + cryst.positions[i])))
      push!(lat_points_rlu,Point3f(collect(lat_loc.I .- 1) + cryst.positions[i]))
      push!(point_sites,i)
    end
    for bz = CartesianIndices(ntuple(i -> nbzs[i],3))
      q = collect(bz.I .- 1) + collect((lat_loc.I .- 1) ./ latsize)
      push!(recip_lat_comm_points,Point3f(q))
      push!(recip_lat_abs_comm_points,Point3f(cryst.recipvecs * q))
    end
  end

  scatter!(ax_real_lattice,lat_points,color = point_sites)
  n(y) = Sunny.number_to_math_string(y;atol = cryst.symprec)
  m(y) = Sunny.number_to_simple_string(y;digits = 4)
  text!(ax_real_lattice,lat_points,color = :black,text = map(x -> "($(m(x[1])),$(m(x[2])),$(m(x[3])))",lat_points),align = (:center,:top),visible = txt_tog.active)

  linesegments!(ax_real_lattice,[(Point3f(-1,0,0),Point3f(1,0,0)),(Point3f(0,-1,0),Point3f(0,1,0)),(Point3f(0,0,-1),Point3f(0,0,1))],color = :black)
  linesegments!(ax_real_lattice,[(Point3f(0,0,0),Point3f(cryst.latvecs[:,j]...)) for j = 1:3],color = [:red,:orange,:magenta],linewidth = 2.5)
  if !isnothing(params)
    linesegments!(ax_real_lattice,[(Point3f(0,0,0),Point3f((cryst.latvecs * params.covectors[j,1:3])...)) for j = 1:3],color = [:blue,:green,:purple],linewidth = 5.5,overdraw = true)
  end

  scatter!(ax_real_lattice,cryst.latvecs[:,1]...,color = :red)
  scatter!(ax_real_lattice,cryst.latvecs[:,2]...,color = :orange)
  scatter!(ax_real_lattice,cryst.latvecs[:,3]...,color = :deeppink)
  text!(ax_real_lattice,cryst.latvecs[:,1]...;text ="a",align = (:right,:bottom),color = :red)
  text!(ax_real_lattice,cryst.latvecs[:,2]...;text ="b",align = (:right,:bottom),color = :orange)
  text!(ax_real_lattice,cryst.latvecs[:,3]...;text ="c",align = (:right,:bottom),color = :deeppink)

  scatter!(ax_real_lattice,rs,0,0,color = :black)
  scatter!(ax_real_lattice,0,rs,0,color = :black)
  scatter!(ax_real_lattice,0,0,rs,color = :black)
  text!(ax_real_lattice,rs,0,0;text ="x (rₛ,Å)",align = (:left,:bottom))
  text!(ax_real_lattice,0,rs,0;text ="y (rₛ,Å)",align = (:left,:bottom))
  text!(ax_real_lattice,0,0,rs;text ="z (rₛ,Å)",align = (:left,:bottom))

  cam0 = Makie.cam3d!(ax_real_lattice.scene;projectiontype = Makie.Orthographic,clipping_mode = :static)

  ax_real_rlu = LScene(f[2,1];show_axis = false)

  M = inv(cryst.latvecs)

  scatter!(ax_real_rlu,lat_points_rlu,color = point_sites)
  text!(ax_real_rlu,lat_points_rlu,color = :black,text = map(x -> "($(n(x[1])),$(n(x[2])),$(n(x[3])))",lat_points_rlu),align = (:center,:top),visible = txt_tog.active)

  linesegments!(ax_real_rlu,[(Point3f(-rs * M[:,j]...),Point3f(rs * M[:,j]...)) for j = 1:3],color = :black)
  linesegments!(ax_real_rlu,[(Point3f(0,0,0),Point3f(I(3)[:,j]...)) for j = 1:3],color = [:red,:orange,:deeppink])

  text!(ax_real_rlu,1,0,0;text ="a",align = (:right,:bottom),color = :red)
  text!(ax_real_rlu,0,1,0;text ="b",align = (:right,:bottom),color = :orange)
  text!(ax_real_rlu,0,0,1;text ="c",align = (:right,:bottom),color = :deeppink)

  scatter!(ax_real_rlu,rs * M[:,1]...,color = :black)
  scatter!(ax_real_rlu,rs * M[:,3]...,color = :black)
  scatter!(ax_real_rlu,rs * M[:,2]...,color = :black)
  text!(ax_real_rlu,rs * M[:,1]...;text ="x (rₛ,Å)",align = (:left,:bottom))
  text!(ax_real_rlu,rs * M[:,2]...;text ="y (rₛ,Å)",align = (:left,:bottom))
  text!(ax_real_rlu,rs * M[:,3]...;text ="z (rₛ,Å)",align = (:left,:bottom))

  if !isnothing(params)
    linesegments!(ax_real_rlu,[(Point3f(0,0,0),Point3f(params.covectors[j,1:3]...)) for j = 1:3],color = [:blue,:green,:purple],linewidth = 5.5,overdraw = true)
  end

  cam1 = Makie.cam3d!(ax_real_rlu.scene;projectiontype = Makie.Orthographic,clipping_mode = :static)

  ax_recip_rlu = LScene(f[2,2];show_axis = false)
  N = cryst.latvecs'

  scatter!(ax_recip_rlu,recip_lat_comm_points,color = [all(isinteger.(x)) ? :red : :black for x in recip_lat_comm_points])
  text!(ax_recip_rlu,recip_lat_comm_points,color = :black,text = map(x -> "($(n(x[1])),$(n(x[2])),$(n(x[3])))",recip_lat_comm_points),align = (:center,:top),visible = txt_tog.active)

  linesegments!(ax_recip_rlu,[(Point3f(-(1/rs) * N[:,j]...),Point3f((1/rs) * N[:,j]...)) for j = 1:3],color = :black)
  linesegments!(ax_recip_rlu,[(Point3f(0,0,0),Point3f(I(3)[:,j]...)) for j = 1:3],color = [:red,:orange,:deeppink])

  text!(ax_recip_rlu,1,0,0;text ="qx [rlu]",align = (:right,:bottom),color = :red)
  text!(ax_recip_rlu,0,1,0;text ="qy [rlu]",align = (:right,:bottom),color = :orange)
  text!(ax_recip_rlu,0,0,1;text ="qz [rlu]",align = (:right,:bottom),color = :deeppink)

  scatter!(ax_recip_rlu,(1/rs) * N[:,1]...,color = :black,markersize = 12)
  scatter!(ax_recip_rlu,(1/rs) * N[:,3]...,color = :black,markersize = 12)
  scatter!(ax_recip_rlu,(1/rs) * N[:,2]...,color = :black,markersize = 12)
  text!(ax_recip_rlu,(1/rs) * N[:,1]...;text ="x (rₛ⁻¹,Å⁻¹)",align = (:left,:bottom))
  text!(ax_recip_rlu,(1/rs) * N[:,2]...;text ="y (rₛ⁻¹,Å⁻¹)",align = (:left,:bottom))
  text!(ax_recip_rlu,(1/rs) * N[:,3]...;text ="z (rₛ⁻¹,Å⁻¹)",align = (:left,:bottom))

  if !isnothing(params)
    mantid_axes_as_columns = inv(params.covectors[1:3,1:3])
    linesegments!(ax_recip_rlu,[(Point3f(0,0,0),Point3f(mantid_axes_as_columns[:,j]...)) for j = 1:3],color = [:blue,:green,:purple],linewidth = 5.5,overdraw = true)
    viz_qqq_path!(ax_recip_rlu,params;bin_colors = [:blue,:green,:purple])#,line_alpha = 1.0,bin_line_width = 1.5)
  end

  cam2 = Makie.cam3d!(ax_recip_rlu.scene;projectiontype = Makie.Orthographic,clipping_mode = :static)

  ax_recip_abs = LScene(f[1,2];show_axis = false)
  L = 2π * I(3)

  scatter!(ax_recip_abs,recip_lat_abs_comm_points,color = [all(isinteger.(x)) ? :red : :black for x in recip_lat_comm_points])
  text!(ax_recip_abs,recip_lat_abs_comm_points,color = :black,text = map(x -> "($(m(x[1])),$(m(x[2])),$(m(x[3])))",recip_lat_abs_comm_points),align = (:center,:top),visible = txt_tog.active)

  linesegments!(ax_recip_abs,[(Point3f(-(1/rs) * L[:,j]...),Point3f((1/rs) * L[:,j]...)) for j = 1:3],color = :black)
  linesegments!(ax_recip_abs,[(Point3f(0,0,0),Point3f(cryst.recipvecs[:,j]...)) for j = 1:3],color = [:red,:orange,:deeppink])

  text!(ax_recip_abs,cryst.recipvecs[:,1]...;text ="qx [rlu]",align = (:right,:bottom),color = :red)
  text!(ax_recip_abs,cryst.recipvecs[:,2]...;text ="qy [rlu]",align = (:right,:bottom),color = :orange)
  text!(ax_recip_abs,cryst.recipvecs[:,3]...;text ="qz [rlu]",align = (:right,:bottom),color = :deeppink)

  scatter!(ax_recip_abs,(1/rs) * L[:,1]...,color = :black,markersize = 12)
  scatter!(ax_recip_abs,(1/rs) * L[:,3]...,color = :black,markersize = 12)
  scatter!(ax_recip_abs,(1/rs) * L[:,2]...,color = :black,markersize = 12)
  text!(ax_recip_abs,(1/rs) * L[:,1]...;text ="x (rₛ⁻¹,Å⁻¹)",align = (:left,:bottom))
  text!(ax_recip_abs,(1/rs) * L[:,2]...;text ="y (rₛ⁻¹,Å⁻¹)",align = (:left,:bottom))
  text!(ax_recip_abs,(1/rs) * L[:,3]...;text ="z (rₛ⁻¹,Å⁻¹)",align = (:left,:bottom))

  if !isnothing(params)
    mantid_axes_as_columns = cryst.recipvecs * inv(params.covectors[1:3,1:3])
    linesegments!(ax_recip_abs,[(Point3f(0,0,0),Point3f(mantid_axes_as_columns[:,j]...)) for j = 1:3],color = [:blue,:green,:purple],linewidth = 5.5,overdraw = true)
    params_abs = copy(params)
    Sunny.bin_absolute_units_as_rlu!(params_abs,cryst)
    viz_qqq_path!(ax_recip_abs,params_abs;bin_colors = [:blue,:green,:purple])#,line_alpha = 1.0,bin_line_width = 1.5)
  end

  cam3 = Makie.cam3d!(ax_recip_abs.scene;projectiontype = Makie.Orthographic,clipping_mode = :static)

  cams = [cam0,cam1,cam2,cam3]
  axs = [ax_real_lattice,ax_real_rlu,ax_recip_rlu,ax_recip_abs]
  mats = Matrix{Matrix}(undef,4,4)
  mats[1,2] = cryst.latvecs
  mats[2,1] = M

  mats[3,1] = M
  mats[1,3] = cryst.latvecs
  mats[2,3] = I(3)
  mats[3,2] = I(3)

  mats[1,4] = cryst.latvecs * inv(cryst.recipvecs)
  mats[4,1] = cryst.recipvecs * inv(cryst.latvecs)
  mats[2,4] = inv(cryst.recipvecs)
  mats[4,2] = cryst.recipvecs
  mats[3,4] = inv(cryst.recipvecs)
  mats[4,3] = cryst.recipvecs
  function update_A_from_B(a,b)
    if a == b
      return # Already up to date!
    end
    camA = cams[a+1]
    camB = cams[b+1]
    mAB = mats[a+1,b+1]
    Observables.setexcludinghandlers!(camA.lookat, mAB * camB.lookat[])
    Observables.setexcludinghandlers!(camA.eyeposition, mAB * camB.eyeposition[])
    Observables.setexcludinghandlers!(camA.upvector, mAB * camB.upvector[])
    update_cam!(axs[a+1].scene)
  end
  function sync_cameras(source)
    for dest = (1:length(cams)) .- 1
      update_A_from_B(dest,source)
    end
  end

  on(x -> sync_cameras(0), cam0.lookat)
  on(x -> sync_cameras(0), cam0.eyeposition)
  on(x -> sync_cameras(1), cam1.lookat)
  on(x -> sync_cameras(1), cam1.eyeposition)
  on(x -> sync_cameras(2), cam2.lookat)
  on(x -> sync_cameras(2), cam2.eyeposition)
  on(x -> sync_cameras(3), cam3.lookat)
  on(x -> sync_cameras(3), cam3.eyeposition)



  f
end
