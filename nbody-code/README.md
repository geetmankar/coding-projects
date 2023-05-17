# Python N-Body Simulation

Use Python $\geq$ 3.9 to avoid unexpected errors.

## CPU version (`nbody_script_cpu.py`)

Uses NumPy (`numpy`)

Run the script to get an N-Body simulation. The script can easily be edited to return a file with the positions of each particle at every time step (by default it does not). The `-sv` or `--save-video` flag will save the results to a 60FPS video.

## GPU version (`nbody_script_gpu.py`)

Uses PyTorch (`torch`)

You can give the argument `-cpu` to the GPU script `nbody_script_gpu` to force the use of the CPU instead. The `-sv` or `--save-video` flag will save the results to a 60FPS video.
