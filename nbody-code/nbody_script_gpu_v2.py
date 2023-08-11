#!/usr/bin/env python3

import argparse
import gc
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from numpy.typing import NDArray
from tqdm.auto import tqdm

# =============================================================


"""
Simple N-body simulation in Python
Based on Newton's Law of Gravity
"""


def parse_args() -> argparse.ArgumentDefaultsHelpFormatter:
    # script arguments
    parser = argparse.ArgumentParser(description="N-body Script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-sv", "--save-video",
        action="store_true",
        help="To save a video of the simulation running"
    )

    parser.add_argument(
        "-cpu", "--cpu",
        action="store_true",
        help="To use the CPU for the simulation"
    )

    return parser.parse_args()

# -----------------------------------------------------------


@dataclass
class NBodySystem:
    """
    N    : Number of Bodies in the system
    G    : Newton's gravitational constant
    soft : length at which to stop calculating the force
           and replacing it with a small value, since Newton's
           gravity becomes infinite as distance between 2 point
           particles tends to zero. Also called softening length.
    mass : N x 1 vector of masses of the N particles
    pos  : N x 3 matrix of positions of the N particles
    vel  : N x 3 matrix of velocities of the N particles
    accel: N x 3 matrix of accelerations of the N particles
    """
    N: int
    G: float
    soft: float
    mass: torch.Tensor
    pos: torch.Tensor
    vel: torch.Tensor
    accel: torch.Tensor


# --------------------------------------------------------

def get_accel(nbsys: NBodySystem) -> torch.Tensor:
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
    inv_r3 = (dx**2 + dy**2 + dz**2 + nbsys.soft**2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)

    # accelerations of the N particles
    ax = nbsys.G * (dx * inv_r3) @ nbsys.mass
    ay = nbsys.G * (dy * inv_r3) @ nbsys.mass
    az = nbsys.G * (dz * inv_r3) @ nbsys.mass

    # @ represents matrix multiplication

    # stack the accelerations
    accel = torch.hstack((ax, ay, az))

    return accel

# --------------------------------------------------------


def get_E(nbsys: NBodySystem) -> tuple[torch.Tensor, torch.Tensor]:
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
    KE = 0.5 * torch.sum(torch.sum(nbsys.mass * nbsys.vel**2))

    # Potential Energy of the system
    # positions of all the particles
    x, y, z = nbsys.pos[:, 0:1], nbsys.pos[:, 1:2], nbsys.pos[:, 2:3]

    # pairwise particle separations: r_j - r_i
    dx, dy, dz = (x.T - x), (y.T - y), (z.T - z)

    # r^-1 for all pairwise particle separations
    inv_r = torch.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r > 0] = 1 / inv_r[inv_r > 0]

    # we sum only over the upper triangle of the matrix
    # to count each pairwise interaction only once
    PE = nbsys.G * torch.sum(
        torch.sum(
            torch.triu(-(nbsys.mass * nbsys.mass.T) * inv_r, 1)
        )
    )

    return KE, PE

# --------------------------------------------------------


def live_plot_nbody(
    particle_positions: torch.Tensor,
    t_end: float,
    t_all: torch.Tensor,
    KE_save: torch.Tensor,
    PE_save: torch.Tensor,
    fig_attrs: tuple[plt.Figure, plt.GridSpec, plt.Axes, plt.Axes],
    sample_dir: Optional[str] = None,
    save_video: bool = False,
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
        xx = particle_positions[:, 0, max(i-10, 0):(i+1)]
        yy = particle_positions[:, 1, max(i-10, 0):(i+1)]
        lines2[0].set_data(xx, yy)
        line_ke.set_data(t_all[:i], KE_save[:i])
        line_pe.set_data(t_all[:i], PE_save[:i])
        return lines + [line_ke, line_pe]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(t_all),
                                   blit=True)

    if save_video:
        print("[Saving Simulation Video...]")
        anim.save(f"{sample_dir}/nbody_gpu.mp4", writer="ffmpeg", fps=60)
        print("[Saved N-Body Simulation Video]")


# --------------------------------------------------------

def run_simulation(
        nbsys: NBodySystem, t_end: float, dt: float, device: torch.device
) -> tuple[NDArray, ...]:
    """
    Runs the simulation for a specified duration.
    ---------------------------
    Arguments:
    ---------------------------
    nbsys     : NBodySystem
    t_end     : Duration of simulation
    dt        : time step for simulation
    ---------------------------
    Returns:
    ---------------------------
    Saves particle positions, velocities, energies at a specified frequency
    """

    # set up arrays to store data
    N_iter = int(np.ceil(t_end/dt))
    pos_save = torch.zeros(nbsys.N, 3, N_iter + 1, device=device)
    # vel_save = torch.zeros(nbody.N, 3, N_iter + 1, device=device)
    KE_save = torch.zeros(N_iter + 1, device=device)
    PE_save = torch.zeros(N_iter + 1, device=device)
    time = torch.zeros(N_iter + 1, device=device)

    # get initial acceleration and energies
    nbsys.accel = get_accel(nbsys)
    KE_save[0], PE_save[0] = get_E(nbsys)

    # save initial conditions
    pos_save[:, :, 0] = nbsys.pos
    # vel_save[:,:,0] = nbody.vel

    # initialize time
    t = 0.0
    i = 0

    for i in tqdm(range(N_iter), unit='steps'):
        # update positions
        nbsys.pos = nbsys.pos + (nbsys.vel * dt) + (0.5 * nbsys.accel * dt**2)
        # find new acceleration
        accel_new = get_accel(nbsys)
        # update velocities
        nbsys.vel = nbsys.vel + 0.5 * (nbsys.accel + accel_new) * dt

        pos_save[:, :, i+1] = nbsys.pos
        # vel_save[:,:,i+1] = nbody.vel
        KE_save[i+1], PE_save[i+1] = get_E(nbsys)
        time[i+1] = t

        # update acceleration
        nbsys.accel = accel_new

        # update time
        t += dt

    # return saved data
    return (pos_save.cpu().numpy(), KE_save.cpu().numpy(),
            PE_save.cpu().numpy(), time.cpu().numpy())

# --------------------------------------------------------


def main():
    config = parse_args()

    N = 1000   # Number of particles
    t_end = 15.    # Time at which the sim ends
    dt = 0.01   # Timestep
    soft = 0.1    # softening length
    G = 3.     # Newton's gravitational constant

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu" if config.cpu else "cpu"
    )

    mass = 20. * torch.ones(N, 1)/N   # total mass of the N particles
    pos = 4 * torch.randn(N, 3)     # random positions
    vel = torch.randn(N, 3)
    accel = torch.zeros(N, 3)

    nbsys = NBodySystem(
        N, G, soft, mass.to(device), pos.to(device),
        vel.to(device), accel.to(device)
    )

    pos, KE_save, PE_save, t_all = run_simulation(nbsys, t_end, dt, device)

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
