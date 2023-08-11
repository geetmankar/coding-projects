#!/usr/bin/env python3

import os
import argparse
import gc
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm.auto import tqdm

# =============================================================


"""
Simple N-body simulation in Python
Based on Newton's Law of Gravity
"""


def parse_args():
    # script arguments
    parser = argparse.ArgumentParser(
        description="N-body Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-sv", "--save-video",
        action="store_true",
        help="To save a video of the simulation running"
    )

    return parser.parse_args()

# -----------------------------------------------------------


@dataclass
class NBodySystem:
    N: int
    G: float
    soft: float
    mass: NDArray
    pos: NDArray
    vel: NDArray
    accel: NDArray

# -----------------------------------------------------------


def get_accel(nbsys: NBodySystem):
    """
    Calculating the acceleration on each particle
    ---------------------------
    Arguments:
    ---------------------------
    nbsys: NBodySystem
    ---------------------------
    Returns:
    ---------------------------
    accel: N x 3 matrix of accelerations for the N particles
    """

    # positions of all the particles
    x, y, z = nbsys.pos[:, 0:1], nbsys.pos[:, 1:2], nbsys.pos[:, 2:3]

    # pairwise particle separations: r_j - r_i
    dx, dy, dz = (x.T - x), (y.T - y), (z.T - z)

    # r^-3 for the pairwise particle separations
    inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + nbsys.soft ** 2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

    # accelerations of the N particles
    ax = nbsys.G * (dx * inv_r3) @ nbsys.mass
    ay = nbsys.G * (dy * inv_r3) @ nbsys.mass
    az = nbsys.G * (dz * inv_r3) @ nbsys.mass

    # @ represents matrix multiplication
    # stack the accelerations
    accel = np.hstack((ax, ay, az))
    return accel


# --------------------------------------------------------

def get_E(nbsys: NBodySystem):
    """
    Get K.E. and P.E. of the simulation
    ---------------------------
    Arguments:
    ---------------------------
    nbsys: NBodySystem
    ---------------------------
    Returns:
    ---------------------------
    KE   : Kinetic Energy of the System
    PE   : Potential Energy of the System
    """

    # Kinetic Energy of the system
    KE = 0.5 * np.sum(np.sum(nbsys.mass * nbsys.vel ** 2))

    # Potential Energy of the system
    # positions of all the particles
    x, y, z = nbsys.pos[:, 0:1], nbsys.pos[:, 1:2], nbsys.pos[:, 2:3]

    # pairwise particle separations: r_j - r_i
    dx, dy, dz = (x.T - x), (y.T - y), (z.T - z)

    # r^-1 for all pairwise particle separations
    inv_r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    inv_r[inv_r > 0] = 1 / inv_r[inv_r > 0]

    # we sum only over the upper triangle of the matrix
    # to count each pairwise interaction only once
    PE = nbsys.G * np.sum(
        np.sum(
            np.triu(-(nbsys.mass * nbsys.mass.T) * inv_r, 1)
        )
    )

    return KE, PE


# --------------------------------------------------------

def live_plot_nbody(
        particle_positions: NDArray,
        t_end: float,
        t_all: NDArray,
        KE_save: NDArray,
        PE_save: NDArray,
        fig_attrs: Tuple[plt.Figure, plt.GridSpec, plt.Axes, plt.Axes],
        save_video: bool = False,
        sample_dir: Optional[str] = None,
) -> None:
    if save_video and sample_dir and not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)

    fig, _grid, ax1, ax2 = fig_attrs
    bticks = list(range(-10, 11, 1))
    # bticks = list(range(-3, 4, 1))

    def ends(lst: list[int]) -> tuple[int, int]:
        return (min(lst), max(lst))

    ax1.set(
        xlabel="$x$", ylabel="$y$",
        xlim=ends(bticks), ylim=ends(bticks),
        xticks=bticks[::2], yticks=bticks[::2],
        aspect='equal',
    )

    ax2.set(
        xlabel="Time", ylabel="Energy",
        xlim=(0, t_end), ylim=(
            min(KE_save.min(), PE_save.min()),
            max(KE_save.max(), PE_save.max())
        ),
    )

    lines = ax1.plot([], [], "o", markersize=1, zorder=3)
    lines2 = ax1.plot([], [], "o", markersize=0.5, alpha=0.5)
    line_ke, = ax2.plot([], [], label="K.E.")
    line_pe, = ax2.plot([], [], label="P.E.")
    ax2.legend(fontsize=6, framealpha=0.5)

    t_all = np.array(t_all)

    def init():
        lines[0].set_data([], [])
        lines2[0].set_data([], [])
        line_ke.set_data([], [])
        line_pe.set_data([], [])
        return lines + [line_ke, line_pe]

    def animate(i):
        pos = particle_positions[:, :, i]
        lines[0].set_data(pos[:, 0], pos[:, 1])
        trails = np.array([i - 50, i - 40, i - 30, i - 20, i - 10, 0])
        trail_len = np.max(trails[np.where(trails >= 0)])
        xx = particle_positions[:, 0, trail_len:(i + 1)]
        yy = particle_positions[:, 1, trail_len:(i + 1)]
        lines2[0].set_data(xx, yy)
        line_ke.set_data(t_all[:i], KE_save[:i])
        line_pe.set_data(t_all[:i], PE_save[:i])
        return lines + [line_ke, line_pe]

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t_all), blit=True
    )

    if save_video:
        print("[Saving Simulation Video]")
        anim.save(f"{sample_dir}/nbody2.mp4", writer="ffmpeg", fps=60)
        print("[Saved N-Body Simulation Video]")

    plt.show()


# --------------------------------------------------------

def run_simulation(
        nbsys: NBodySystem, t_end: float, dt: float
    ) -> tuple[NDArray, ...]:
    """
    Runs the simulation for a specified duration.
    ---------------------------
    Arguments:
    ---------------------------
    nbsys     : N-Body System
    t_end     : Duration of simulation
    dt        : time step for simulation
    ---------------------------
    Returns:
    ---------------------------
    Saves particle positions, velocities, energies at a specified frequency
    """

    # set up arrays to store data
    N_iter = int(np.ceil(t_end / dt))
    pos_save = np.zeros((nbsys.N, 3, N_iter + 1))
    KE_save = np.zeros(N_iter + 1)
    PE_save = np.zeros(N_iter + 1)
    time = np.zeros(N_iter + 1)

    # get initial acceleration and energies
    nbsys.accel = get_accel(nbsys)
    KE_save[0], PE_save[0] = get_E(nbsys)

    # save initial conditions
    pos_save[:, :, 0] = nbsys.pos

    # initialize time
    t = 0.0

    for i in tqdm(range(N_iter), unit='steps'):
        # update positions
        nbsys.pos = nbsys.pos + (nbsys.vel * dt) + \
            (0.5 * nbsys.accel * dt ** 2)

        # find new acceleration
        accel_new = get_accel(nbsys)

        # update velocities
        nbsys.vel = nbsys.vel + 0.5 * (nbsys.accel + accel_new) * dt

        pos_save[:, :, i + 1] = nbsys.pos
        # vel_save[:,:,i+1] = nbody.vel
        KE_save[i + 1], PE_save[i + 1] = get_E(nbsys)
        time[i + 1] = t

        # update acceleration
        nbsys.accel = accel_new
        # update time
        t += dt

    # return saved data
    return pos_save, KE_save, PE_save, time


# --------------------------------------------------------

def main():
    config = parse_args()

    N = 1000  # Number of particles
    t_end = 15.  # Time at which the sim ends
    dt = 0.01  # Timestep
    soft = 0.1  # softening length
    G = 3.  # Newton's gravitational constant

    rng = np.random.default_rng(seed=42)
    mass = 20. * np.ones((N, 1)) / N  # total mass of the N particles
    pos = 4 * rng.standard_normal((N, 3))  # random positions
    vel = rng.standard_normal((N, 3))
    accel = np.zeros((N, 3))

    nbsys = NBodySystem(N, G, soft, mass, pos, vel, accel)

    pos, KE_save, PE_save, t_all = run_simulation(nbsys, t_end, dt)

    fig = plt.figure(figsize=(4, 6), dpi=100)
    grid = plt.GridSpec(3, 1, wspace=0, hspace=0.7)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])

    fig_attrs = (fig, grid, ax1, ax2)

    live_plot_nbody(
        particle_positions=pos,
        t_end=t_end, t_all=t_all,
        KE_save=KE_save, PE_save=PE_save,
        fig_attrs=fig_attrs,
        sample_dir='./data',
        save_video=config.save_video,
    )


# --------------------------------------------------------

if __name__ == "__main__":
    main()
    gc.collect()
