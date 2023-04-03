# How to run the script

## Change the permissions

`chmod +x ./%FILE_PATH%/nbody_script.py`

## Then run the script:

### In console (not a Jupyter Notebook or Colab)

`./%FILE_PATH%/nbody_script.py`

### In a Jupyter Notebook or Colab Notebook

`./%FILE_PATH%/nbody_script.py [-ip or --ipynb]` using the shell magic `%%sh` or `!` in a cell.

OR

`%run ./%FILE_PATH%/nbody_script.py [-ip or --ipynb]` in a cell.

---

`python_requirements.txt` contains the names of the required packages. The name of the package `ipython` can be removed from `python_requirements.txt` before installing if you won't use the script in a Notebook environment.

By the way the command for installing packages using a requirements text file is:

`pip3 install -r python_requirements.txt`
