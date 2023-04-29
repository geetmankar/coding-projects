#!/usr/bin/env python3

import os, argparse, gc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, notebook


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
        help="If running script in IPython Notebook"
    )

    parser.add_argument(
        "-sv", "--save-video",
        action="store_true",
        help="If running script in IPython Notebook"
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

def live_plot_nbody(**kwargs):
    
    if kwargs["save_video"] and not os.path.isdir(kwargs["sample_dir"]):
        os.makedirs(kwargs["sample_dir"])

    xx, yy, pos = kwargs["xx"], kwargs["yy"], kwargs["pos"]
    t_end, t_all= kwargs["t_end"], kwargs["t_all"]
    KE_save, PE_save = kwargs["KE_save"], kwargs["PE_save"]

    # fig  = plt.figure(figsize=(4,5), dpi=100)
    # grid = plt.GridSpec(3, 1, wspace=0, hspace=0.7)
    # ax1  = plt.subplot(grid[0:2, 0])
    # ax2  = plt.subplot(grid[  2, 0])

    fig, grid, ax1, ax2 = kwargs["fig_attrs"]

    
    ax1.set(
        xlabel="$x$", ylabel="$y$", xlim=(-2, 2), ylim=(-2, 2),
        xticks = [-2,-1,0,1,2], yticks = [-2,-1,0,1,2],
    )
    ax1.set_aspect('equal', 'box')
    ax1.scatter(xx, yy, s=1, color=[.7, .7, 1]) # trails
    ax1.scatter(pos[:,0], pos[:,1],s=5, color='blue') # current pos

    ax2.set(xlabel='Time', ylabel='Energy', xlim=(0, t_end), ylim=(-300, 300))
    ax2.set_aspect(0.007)
    ax2.scatter(t_all, KE_save        , color='red'  ,s=0.5, label='KE'   )
    ax2.scatter(t_all, PE_save        , color='aqua' ,s=0.5, label='PE'   )
    ax2.scatter(t_all, KE_save+PE_save, color='black',s=0.5, label='E_tot')
    ax2.legend(loc='upper right', ncol=3, frameon=False, fontsize=7)

    plt.sca(plt.gca())
    if kwargs["save_video"]:
        plt.savefig(kwargs["sample_dir"] + f'/nbody_{i:04d}.png',
                    dpi=240, bbox_inches='tight')
    

    if kwargs["ipynb"]:
        # ax1.clear(); ax2.clear()
        fig.canvas.draw()
        fig.show()
        import IPython.display as display
        display.display(fig)
        display.clear_output(wait=True)

    plt.pause(0.001)

    if not kwargs["ipynb"]:
        ax1.clear(); ax2.clear()
    
###########--------------------------------------------------------

def main():

    """ Run the N-body simulation """
    config = parse_args()
    ipynb = config["ipynb"]
    save_video = config["save_video"]

    if ipynb: 
        import IPython.display as display
    
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


    print("Starting simulation loop and plotting") ###################
    # Simulation Loop
    if ipynb:
        rangetqdm = notebook.tqdm(iterable=range(Nt), leave=True)
    else:
        rangetqdm = tqdm(range(Nt))

    # calculation and plotting loop
    fig  = plt.figure(figsize=(4,5), dpi=100)
    grid = plt.GridSpec(3, 1, wspace=0, hspace=0.7)
    ax1  = plt.subplot(grid[0:2, 0])
    ax2  = plt.subplot(grid[  2, 0])
    
    # if ipynb: plt.ion(); fig.show(); fig.canvas.draw()

    for i in rangetqdm:
        
        vel += acc * dt/2.  # half kick
        pos += vel * dt     # positional drift
        # update accelerations for the particles
        acc  = get_accel(pos, mass, G, soft) 
        vel += acc * dt/2.  # half kick
        t   += dt             # update time
        KE, PE = get_E(pos, vel, mass, G) # get energy of the system

        # save energies and positions again
        pos_arr[:, :, i+1] = pos
        KE_save[i+1] = KE
        PE_save[i+1] = PE

        # Real Time Plotting Loop
        if plotRealTime or (i == Nt-1):
            
            xx = pos_arr[:, 0, np.max(i-50, 0):(i+1)]
            yy = pos_arr[:, 1, np.max(i-50, 0):(i+1)]
            
            if ipynb: display.display(rangetqdm.container)

            live_plot_nbody(
                xx=xx, yy=yy, pos=pos,
                t_end=t_end, t_all=t_all,
                KE_save=KE_save, PE_save=PE_save,
                fig_attrs = (fig, grid, ax1, ax2),
                save_video=save_video, sample_dir=sample_dir,
                ipynb=ipynb,
                )
            

    print(f"Simulation Finished for {N} particles") ###################
    
    # Save figure
    if save_video:
        import cv2
        print("Creating the video")
        
        vid_fname = 'Nbody_plots.avi'

        files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'nbody' in f]
        files.sort()

        img1 = cv2.imread(sample_dir + "/nbody_0000.png")
        img1sh = img1.shape[:2][::-1]

        out = cv2.VideoWriter(vid_fname, cv2.VideoWriter_fourcc(*'MP4V'), 60, img1sh)
        [out.write(cv2.imread(fname)) for fname in files]
        out.release()



###########--------------------------------------------------------

if __name__=="__main__":
    main()
    gc.collect()
    