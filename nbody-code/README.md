# Python N-Body Simulation

## CPU version

Run the script to see a live plot of an N-Body simulation. Add the `-sv` or `--save-video` option with the script to write a video file using the `cv2` package. Do NOT use the `-ip` or `--ipynb` option as it is still not fixed, hence the script does not show live plots in notebooks properly. Use Python $\geq$ 3.9 to avoid unexpected errors.

## GPU version

The GPU version does not require the `cv2` package for writing the video. You can give the argument `-cpu` to the GPU script `nbody_script_gpu` to force the use of the CPU instead. `-sv` or `--save-video` flag will save the results to a 60FPS video.
