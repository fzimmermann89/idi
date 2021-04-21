
CAVE: Hic sunt dracones

_The code is a mess, undocumented and only certain code paths are tested._


# IDI - INCOHERENT DIFFRACTION IMAGING

[![https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg](https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg)](https://singularity-hub.org/collections/4824)
![tests](https://github.com/fzimmermann89/idi/actions/workflows/test.yml/badge.svg) 
![Build Status](https://www.travis-ci.com/fzimmermann89/idi.svg?branch=master)

Singularity Image now at [library://fzimmermann89/idi/idi](https://cloud.sylabs.io/library/_container/607b669a4ad4aa1fdea0c43c)

Conda Pacakges at [zimmf/idi](https://anaconda.org/zimmf/idi)

PIP Source at [idi](https://pypi.org/project/idi/)

Wheels at [Releases](https://github.com/fzimmermann89/idi/releases/latest)

content of the repo   
------------
- ipynb: example notebooks
- simulation: simulation of incoherent images
- reconstruction: direct and ft based reconstruction
- util: some small utilities for data analysis, geometry and random distributions, etc.


preparation for slac sdf:
---------------------------
Use Singulariy, if using OOD launcher, use the following to start a jupyterhub

```
    function jupyter() { singularity run --app jupyter --nv -B /sdf,/gpfs,/scratch,/lscratch library://fzimmermann89/idi/idi $@; }
```


preparation for sacla:
---------------------------
- Download and install miniconda, setup ssh tunnel for web access.
- `conda create -n local3 python=3.7 numpy mkl mkl-dev ipython ipykernel cython jinja2 numba numexpr matplotlib six scipy jupyterlab`
- `conda activate local3`
- `pip install https://github.com/fzimmermann89/idi/`
- `python -m ipykernel install --user --name local-simulation-env3 --display-name "local simulation(py37)"`




(C) Felix Zimmermann
