#!/usr/bin/bash

echo "Making video from images..."
ffmpeg -framerate 60 -pattern_type  glob -i "images/nbsys_*.png" nbody.mp4 -y >/dev/null 2>&1
echo "Video created!"