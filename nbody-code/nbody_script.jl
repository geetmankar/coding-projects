#!/usr/bin/env julia

using ArgParse
using LinearAlgebra
using Plots
using ProgressMeter
using Random
using Statistics
using VideoIO


# using DocumentFormat
# using JuliaFormatter
###=============================================================

"""
Simple N-body simulation in Julia
Based on Newton's Law of Gravity
"""

function parse_cl()
    # script arguments
    commands = ArgParseSettings(description = "A Julia NBody Script")
    @add_arg_table commands begin
        (["--save-video", "-v"];
        help = "To save a video of the simulation running";
        action = :store_true)
    end

    return parse_args(commands)
end

###-----------------------------------------------------------

mutable struct NBodySystem
    N::Int
    G::Float64
    soft::Float64
    mass::Matrix{Float64}
    pos::Matrix{Float64}
    vel::Matrix{Float64}
    accel::Matrix{Float64}
end

###########--------------------------------------------------------

"""
Calculating the acceleration on each particle

Arguments:
nbsys::NBodySystem
---------------------------

Returns:
accel: N x 3 matrix of accelerations for the N particles
---------------------------
"""
function get_accel(nbsys::NBodySystem)
    x, y, z = nbsys.pos[:, 1], nbsys.pos[:, 2], nbsys.pos[:, 3]
    dx, dy, dz = x' .- x, y' .- y, z' .- z
    inv_r3 = @. sqrt(dx^2 + dy^2 + dz^2 + nbsys.soft^2)
    inv_r3[inv_r3.>0] .= inv_r3[inv_r3.>0] .^ (-1.5)

    ax = (nbsys.G .* dx .* inv_r3) * nbsys.mass
    ay = (nbsys.G .* dy .* inv_r3) * nbsys.mass
    az = (nbsys.G .* dz .* inv_r3) * nbsys.mass

    return hcat(ax, ay, az)
end

###########--------------------------------------------------------
"""
Get K.E. and P.E. of the simulation
---------------------------
Arguments:
nbsys::NBodySystem
---------------------------
Returns:
KE : Kinetic Energy of the System
PE : Potential Energy of the System
"""
function get_E(nbsys::NBodySystem)
    KE = 0.5 * sum(sum(nbsys.mass .* nbsys.vel .^ 2))

    x, y, z = nbsys.pos[:, 1], nbsys.pos[:, 2], nbsys.pos[:, 3]
    dx, dy, dz = (x' .- x), (y' .- y), (z' .- z)
    inv_r3 = @. sqrt(dx^2 + dy^2 + dz^2 + nbsys.soft^2)
    inv_r3[inv_r3.>0] .= inv_r3[inv_r3.>0] .^ (-1)

    PE = nbsys.G .* sum(
        sum(
            triu(-(nbsys.mass .* nbsys.mass') .* inv_r3, 1),
        ),
    )
    return KE, PE
end

###########--------------------------------------------------------

function plot_nbody(
    particle_positions::Array{Float64, 3},
    t_end::Float64,
    t_all::Vector{Float64},
    KE_save::Vector{Float64},
    PE_save::Vector{Float64};
    sample_dir::String = "data",
    save_video::Bool = false,
)::Nothing
    if save_video && !isdir(sample_dir)
        mkdir(sample_dir)
    end
    
    function plot_func(i)

        pos = particle_positions[:, :, i]
        x, y = pos[:, 1], pos[:, 2]
        trail_len = max(1, i - 10)
        xx = particle_positions[:, 1, trail_len:i]
        yy = particle_positions[:, 2, trail_len:i]
        ke = (t_all[1:i], KE_save[1:i])
        pe = (t_all[1:i], PE_save[1:i])

        fig1 = scatter(
            x, y;
            aspect_ratio = :equal,
            markersize = 3, markeralpha = 1, markerstrokewidth = 0,
            xlabel = "X", ylabel = "Y", xlims = (-2, 2), ylims = (-2, 2),
            xticks = [-2, -1, 0, 1, 2], yticks = [-2, -1, 0, 1, 2],
            legend = false,
        )

        scatter!(
            fig1, xx, yy; ms = 1, ma = 0.2,
        )

        fig2 = scatter(
            [ke, pe];
            xlabel = "Time", ylabel = "Energy", markerstrokewidth = 0,
            xlims = (0, t_end), ylims = (min(minimum(KE_save), minimum(PE_save)),
                max(maximum(KE_save), maximum(PE_save))),
            labels = ["K.E." "P.E."], legend = true, fg_legend = :transparent,
        )
        plot(
            fig1, fig2,
            size=(502, 700),
            layout = grid(2, 1, heights = [0.9, 0.1], widths=[1.0, 0.6]),
            framestyle = [:box :box],
        )
    end

    pbar = Progress(length(t_all), desc = "Plotting...")
    anim = @animate for i in 1:length(t_all)
        plot_func(i)
        next!(pbar)
    end

    if save_video
        filename = joinpath(sample_dir, "nbody_jl.mp4")
        mp4(anim, filename, fps = 60)
    end

    return nothing
end


###########--------------------------------------------------------
"""
Runs the simulation for a specified duration.
---------------------------
Arguments:
---------------------------
nbody     : Nbody class object
t_end     : Duration of simulation
dt        : time step for simulation
---------------------------
Returns:
---------------------------
Saves particle positions, velocities, energies at a specified frequency
"""
function run_simulation(nbsys::NBodySystem, t_end::Float64, dt::Float64)
    N = nbsys.N

    # set up arrays to store data
    N_iter = ceil(Int, t_end / dt)
    pos_save = zeros(N, 3, N_iter + 1)
    KE_save = zeros(N_iter + 1)
    PE_save = zeros(N_iter + 1)
    time = zeros(N_iter + 1)

    # get initial acceleration and energies
    nbsys.accel = get_accel(nbsys)
    KE, PE = get_E(nbsys)

    # save initial conditions
    pos_save[:, :, 1] .= nbsys.pos
    KE_save[1] = KE
    PE_save[1] = PE

    # initialize time
    t = 0.0

    pbar = Progress(N_iter, desc = "Running simulation...")
    for i in 1:N_iter
        nbsys.pos .+= (nbsys.vel .* dt) .+ (0.5 .* nbsys.accel .* dt^2)
        accel_new = get_accel(nbsys)
        nbsys.vel .+= 0.5 .* (nbsys.accel .+ accel_new) .* dt

        pos_save[:, :, i+1] .= nbsys.pos
        KE, PE = get_E(nbsys)
        KE_save[i+1] = KE
        PE_save[i+1] = PE
        time[i+1] = t

        nbsys.accel = accel_new
        t += dt
        next!(pbar)
    end

    # return saved data
    return pos_save, KE_save, PE_save, time
end

###########--------------------------------------------------------

function main()::Nothing
    config = parse_cl()
    save_video = config["save-video"]

    N = 100      # Number of particles
    t_end = 10.0 # Time at which the sim ends
    dt = 0.01    # Timestep
    soft = 0.1   # softening length
    G = 1.0      # Newton's gravitational constant

    rng = Random.MersenneTwister(100)
    mass = 20.0 * ones(Float64, N, 1) / N   # total mass of the N particles
    pos = randn(rng, N, 3)     # random positions
    vel = randn(rng, N, 3)
    accel = zeros(Float64, N, 3)

    nbsys = NBodySystem(N, G, soft, mass, pos, vel, accel)

    pos, KE_save, PE_save, t_all = run_simulation(nbsys, t_end, dt)

    plot_nbody(pos, t_end, t_all, KE_save, PE_save;
        sample_dir = "data", save_video = save_video)

end

###########--------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    gr()
    main()
end
