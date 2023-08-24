# Gravitational N-Body Simulation
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Julia](https://img.shields.io/badge/-Julia-9558B2?style=for-the-badge&logo=julia&logoColor=white)
![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)

## Rust
All relevant Rust code is inside the [nbody-rs](https://github.com/geetmankar/coding-projects/tree/main/nbody-code/nbody-rs) folder.

**REQUIRES: A Linux OS (or WSL2), Python 3, matplotlib, ffmpeg CLI tool**

Install the dependencies required for plotting by:
```shell
sudo apt-get install liblapacke-dev libopenblas-dev python3-pip
pip install matplotlib
```
Bonus: Install `sccache` crate (`cargo install sccache` in the terminal) so that you **don't** have to compile all libraries everytime you run the project, only the parts that have changed do.

1. Install **Rust** (see [rustup.rs](rustup.rs)) and enter the [nbody-rs](https://github.com/geetmankar/coding-projects/tree/main/nbody-code/nbody-rs) directory.
1. Open the terminal in this directory, and type: ```cargo run -- -v```. This will run the simulation and save the animation video for the N-Body System.

The `-v` or `--video` flag is to indicate you want to create a video, the images of the plots (from which the video is made) will be subsequently deleted. If you wish to keep both the images and the videos, use the `-i` or `--images` flag along with the video flag.

**Keep in mind that if you do not provide any flags, the images will be saved but NO VIDEO will be created.** If you wish to create a video from the images yourself, use `ffmpeg` (install it using `sudo apt install ffmpeg`) as follows:
```shell
ffmpeg -framerate 60 -pattern_type  glob -i "images/nbsys_*.png" nbody.mp4
```

---

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


