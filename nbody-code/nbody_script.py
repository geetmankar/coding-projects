#!/usr/bin/env python3

import os, argparse, gc
import numpy as np
import matplotlib.pyplot as plt
import cv2
from IPython.display import display, clear_output
from tqdm import tqdm, tqdm_notebook

"""
Simple N-body simulation in Python
Based on Newton's Law of Gravity
"""
def parse_args():
    #script arguments
    parser = argparse.ArgumentParser(description="N-body Script",
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-ip", "--ipynb",
        action="store_true",
        help="If running script in IPython"
    )

    config = vars(parser.parse_args())

    return config


def get_accel(pos: np.ndarray, mass, G: float, soft):
    """
    Calculating the acceleration on each particle
    ---------------------------
    Arguments:
    ---------------------------
    pos  : N x 3 matrix of positions of the N particles
    mass : N x 1 vector of masses of the N particles
    G    : Newton's gravitational constant
    soft : length at which to stop calculating the force
           and replacing it with a small value, since Newton's
           gravity becomes infinite as distance between 2 point
           particles tends to zero. Also called softening length.
    ---------------------------
    Returns:
    ---------------------------
    accel: N x 3 matrix of accelerations for the N particles
    """

    # positions of all the particles
    x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]

    # pairwise particle separations: r_j - r_i
    dx, dy, dz = (x.T - x), (y.T - y), (z.T - z)

    # r^-3 for the pairwise particle separations
    inv_r3 = (dx**2 + dy**2 + dz**2 + soft**2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)

    # accelerations of the N particles
    ax, ay, az = G*(dx*inv_r3)@mass, G*(dy*inv_r3)@mass, G*(dz*inv_r3)@mass
    # @ represents matrix multiplication

    # stack the accelerations
    accel = np.hstack((ax, ay, az))

    return accel

###########--------------------------------------------------------

def get_E(pos, vel, mass, G):
    """
    Get K.E. and P.E. of the simulation
    ---------------------------
    Arguments:
    ---------------------------
    pos  : N x 3 matrix of positions of the N particles
    vel  : N x 3 matrix of velocities of the N particles
    mass : N x 1 vector of masses of the N particles
    G    : Newton's gravitational constant
    ---------------------------
    Returns:
    ---------------------------
    KE   : Kinetic Energy of the System
    PE   : Potential Energy of the System
    """

    # Kinetic Energy of the system
    KE = 0.5 * np.sum(np.sum( mass * vel**2 ))

    # Potential Energy of the system
    # positions of all the particles
    x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]

    # pairwise particle separations: r_j - r_i
    dx, dy, dz = (x.T - x), (y.T - y), (z.T - z)

    # r^-1 for all pairwise particle separations
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r > 0] = 1 / inv_r[inv_r > 0]

    # we sum only over the upper triangle of the matrix
    # to count each pairwise interaction only once
    PE = G * np.sum(
                    np.sum(
                           np.triu( -(mass*mass.T)*inv_r, 1 )
                            )
                    )
    
    return KE, PE

###########--------------------------------------------------------

def main():

    """ Run the N-body simulation """
    config = parse_args()
    ipynb = config["ipynb"]

    sample_dir = "./data"
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    #hasattr(__builtins__,'__IPYTHON__')
    # Simulation parameters 
    N             = 100    # Number of particles
    t             = 0      # Initial time of the sim
    t_end         = 10.    # Time at which the sim ends
    dt            = 0.01   # Timestep
    soft          = 0.1    # softening length
    G             = 1.     # Newton's gravitational constant
    plotRealTime  = True   # Plot at each timestep

    print(f"Simulation inititated for {N} particles") ###################
    # Initial condition
    np.random.seed(15)      # set the random generator seed

    mass = 20. * np.ones((N,1))/N   # total mass of the N particles
    pos  = np.random.randn(N,3)     # random positions
    vel  = np.random.randn(N,3)     # random velocities

    # convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate accelerations
    acc = get_accel(pos, mass, G, soft)

    # calculate Initial energy of the system
    KE, PE = get_E(pos, vel, mass, G)

    # no. of timesteps
    Nt = int(np.ceil( t_end/dt ))

    # save positions (for potting trails) and energies
    pos_arr            = np.zeros(( N, 3, Nt+1 ))
    pos_arr[:, :, 0]   = pos
    KE_save            = np.zeros(Nt+1)
    PE_save            = np.zeros(Nt+1)
    KE_save[0]         = KE
    PE_save[0]         = PE
    t_all              = np.arange(Nt+1) * dt

    # prep to plot figure
    fig  = plt.figure(figsize=(4,5), dpi=100)
    grid = plt.GridSpec(3, 1, wspace=0, hspace=0.3)
    ax1  = plt.subplot(grid[0:2, 0])
    ax2  = plt.subplot(grid[  2, 0])

    print("Starting simulation loop and plotting") ###################
    # Simulation Loop
    if ipynb:
        rangetqdm = tqdm_notebook(iterable=range(Nt), leave=True)
    else:
        rangetqdm = tqdm(range(Nt))
    for i in rangetqdm:
        
        vel += acc * dt/2.  # half kick
        pos += vel * dt     # positional drift
        # update accelerations for the particles
        acc = get_accel(pos, mass, G, soft) 
        vel += acc * dt/2.  # half kick
        t += dt             # update time
        KE, PE = get_E(pos, vel, mass, G) # get energy of the system

        # save energies and positions again
        pos_arr[:, :, i+1] = pos
        KE_save[i+1] = KE
        PE_save[i+1] = PE

        # Real Time Plotting Loop
        if plotRealTime or (i == Nt-1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_arr[:, 0, np.max(i-50, 0):(i+1)]
            yy = pos_arr[:, 1, np.max(i-50, 0):(i+1)]
            plt.scatter(xx, yy, s=1, color=[.7, .7, 1])
            plt.scatter(pos[:,0],pos[:,1],s=10,color='blue')
            ax1.set(xlim=(-2, 2), ylim=(-2, 2))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-2,-1,0,1,2])
            ax1.set_yticks([-2,-1,0,1,2])

            plt.sca(ax2)
            plt.xlabel('Time')
            plt.ylabel('Energy')
            plt.cla()
            plt.scatter(t_all, KE_save        , color='red'  , s=1, label='KE'   )
            plt.scatter(t_all, PE_save        , color='blue' , s=1, label='PE'   )
            plt.scatter(t_all, KE_save+PE_save, color='black', s=1, label='E_tot')
            ax2.set(xlim=(0, t_end), ylim=(-300, 300))
            ax2.set_aspect(0.007)
            ax2.legend(loc='upper right', ncol=3)
            plt.savefig(sample_dir + f'/nbody_{i:04d}.png', dpi=240,
                        bbox_inches='tight')
            
            # check if running in IPython Notebooks
            if ipynb:
                clear_output(wait=True)
                display(rangetqdm.container)
                display(fig)
            else:
                plt.show()
            
            plt.pause(1e-3)

    print(f"Simulation Finished for {N} particles") ###################
    
    # Save figure
    print("Creating the video")
    
    vid_fname = 'Nbody_plots.avi'

    files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'nbody' in f]
    files.sort()

    img1 = cv2.imread(sample_dir + "/nbody_0000.png")
    img1sh = img1.shape[:2][::-1]

    out = cv2.VideoWriter(vid_fname, cv2.VideoWriter_fourcc(*'MP4V'), 60, img1sh)
    [out.write(cv2.imread(fname)) for fname in files]
    out.release()


    return 0


###########--------------------------------------------------------

if __name__=="__main__":
    main()
    gc.collect()
    
