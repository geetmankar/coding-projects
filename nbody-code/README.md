# Python N-Body Simulation

Use Python $\geq$ 3.9 to avoid unexpected errors.

## CPU version (`nbody_script_cpu.py`)

Uses NumPy (`numpy`)

Run the script to get an N-Body simulation. The `-sv` or `--save-video` flag will save the results to a 60FPS video (the `-sv` flag is highly recommended). MAKE SURE TO HAVE `ffmpeg` INSTALLED TO SAVE THE VIDEO (otherwise change the extension of `nbody.mp4` to `nbody.avi` in the script).

## GPU version (`nbody_script_gpu.py`)

Uses PyTorch (`torch`)

You can give the argument `-cpu` to the GPU script `nbody_script_gpu` to force the use of the CPU instead. Flags are similar to the CPU version.

https://github.com/geetmankar/coding-projects/assets/27027921/2df3b741-c699-42f6-8ff3-a1e95dc4e4b1

