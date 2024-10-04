## Install dependencies

### (Recommended) Install with conda:

1. Install conda, if you don't already have it. Miniconda is a lightweight version of conda that is generally easier to use: [https://docs.conda.io/projects/miniconda/en/latest/](https://docs.conda.io/projects/miniconda/en/latest/).


This install will modify the `PATH` variable in your bashrc.
You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

2. Create a conda environment that will contain Python 3:
```
conda create -n cs285 python=3.9
```

3. Activate the environment (do this every time you open a new terminal and want to run code):
```
source activate cs285
```

4. Install the requirements into this conda environment
```
pip install -r requirements.txt
pip install numpy==1.19.5
```

5. Allow your code to be able to see the homework.
```
pip install -e .
```

This conda environment requires activating it every time you open a new terminal (in order to run code), but the benefit is that the required dependencies for this codebase will not affect existing/other versions of things on your computer. This stand-alone environment will have everything that is necessary.



## Troubleshooting 

You may encounter the following GLFW errors if running on machine without a display:

GLFWError: (65544) b'X11: The DISPLAY environment variable is missing'
  warnings.warn(message, GLFWError)
GLFWError: (65537) b'The GLFW library is not initialized'

These can be resolved with:
```
export MUJOCO_GL=egl
```