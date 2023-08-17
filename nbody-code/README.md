# Gravitational N-Body Simulation
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Julia](https://img.shields.io/badge/-Julia-9558B2?style=for-the-badge&logo=julia&logoColor=white)
![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)

## Rust [UNDER CONSTRUCTION]
All relevant Rust code is inside the [nbody-rs](https://github.com/geetmankar/coding-projects/tree/main/nbody-code/nbody-rs) folder. Instructions for the rust version will be added once it is complete.


## Python
Use Python $\geq$ 3.9 to avoid unexpected errors. Run `pip install -r requirements.txt` in your terminal to install the packages needed to run BOTH scripts.

### CPU version (`nbody_script_cpu.py`)

Uses NumPy (`numpy`) for the calculations

Run the script to get an N-Body simulation. The `-sv` or `--save-video` flag will save the results to a 60FPS video (the `-sv` flag is highly recommended). MAKE SURE TO HAVE `ffmpeg` INSTALLED TO SAVE THE VIDEO (otherwise change the extension of `nbody.mp4` to `nbody.avi` in the script).

### GPU version (`nbody_script_gpu.py`)

Uses PyTorch (`torch`) for the calculations

You can give the argument `-cpu` to the GPU script `nbody_script_gpu` to force the use of the CPU instead. Flags are same as the CPU version.

### Version 2 (`*v2.py`)

Same as the above codes but with $1000$ particles. This more clearly shows the performance difference between the CPU and GPU versions of the code. This version also takes greater advantage of the DataClass `NBodySystem` to keep track of the whole system.

The resulting video will look like:



https://github.com/geetmankar/coding-projects/assets/27027921/04ed85e1-cc7d-4ee8-8392-473dfbf63646



---

## Julia
Use Julia $\geq$ 1.9.2 to avoid unexpected errors. Run `julia packages.jl` in your terminal to install the packages needed.

Run the script to get an N-Body simulation by running `julia nbody_script.jl`. The `-v` or `--save-video` flag will save the results to a 60FPS video (the `-v` flag is highly recommended), i.e., run `julia nbody_script.jl -v`. The resulting video will look like:

https://github.com/geetmankar/coding-projects/assets/27027921/173ff342-ec93-4ce7-ac76-451d4c579de9


