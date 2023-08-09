# Gravitational N-Body Simulation

## Python
Use Python $\geq$ 3.9 to avoid unexpected errors. Run `pip install -r requirements.txt` in your terminal to install the packages needed to run BOTH scripts.

### CPU version (`nbody_script_cpu.py`)

Uses NumPy (`numpy`) for the calculations

Run the script to get an N-Body simulation. The `-sv` or `--save-video` flag will save the results to a 60FPS video (the `-sv` flag is highly recommended). MAKE SURE TO HAVE `ffmpeg` INSTALLED TO SAVE THE VIDEO (otherwise change the extension of `nbody.mp4` to `nbody.avi` in the script).

### GPU version (`nbody_script_gpu.py`)

Uses PyTorch (`torch`) for the calculations

You can give the argument `-cpu` to the GPU script `nbody_script_gpu` to force the use of the CPU instead. Flags are same as the CPU version.

The resulting video will look like:



https://github.com/geetmankar/coding-projects/assets/27027921/04ed85e1-cc7d-4ee8-8392-473dfbf63646



---

## Julia
Use Julia $\geq$ 1.9.2 to avoid unexpected errors. Run `julia packages.jl` in your terminal to install the packages needed.

Run the script to get an N-Body simulation by running `julia nbody_script.jl`. The `-v` or `--save-video` flag will save the results to a 60FPS video (the `-v` flag is highly recommended), i.e., run `julia nbody_script.jl -v`. The resulting video will look like:

https://github.com/geetmankar/coding-projects/assets/27027921/173ff342-ec93-4ce7-ac76-451d4c579de9


