
# IDI - INCOHERENT DIFFRACTION IMAGING 🐉

![tests](https://github.com/fzimmermann89/idi/actions/workflows/test.yml/badge.svg)

Singularity Image now at [library://fzimmermann89/idi/idi](https://cloud.sylabs.io/library/_container/607b669a4ad4aa1fdea0c43c)

Conda Pacakges at [zimmf/idi](https://anaconda.org/zimmf/idi)

PIP Source and Wheels at [idi](https://pypi.org/project/idi/)

Wheels at [Releases](https://github.com/fzimmermann89/idi/releases/latest)

content of the repo
-------------------

- ipynb: example notebooks
- simulation: simulation of incoherent images
- reconstruction: direct and ft based reconstruction
- util: some small utilities for data analysis, geometry and random distributions, etc.

installation
------------

- for gpu support you need to install a cupy version matching your cuda version
- afterwards, pip install 'git+<https://github.com/fzimmermann89/idi>' to get the latest version

preparation for slac sdf
------------------------

Use Singulariy, if using OOD launcher, use the following to start a jupyterhub

```
    function jupyter() { singularity run --app jupyter --nv -B /sdf,/gpfs,/scratch,/lscratch library://fzimmermann89/idi/idi $@; }
```

preparation for sacla
---------------------

- Download and install miniconda, setup ssh tunnel for web access.
- `conda create -n local3 python=3.7 numpy mkl mkl-dev ipython ipykernel cython jinja2 numba numexpr matplotlib six scipy jupyterlab`
- `conda activate local3`
- `pip install https://github.com/fzimmermann89/idi/`
- `python -m ipykernel install --user --name local-simulation-env3 --display-name "local simulation(py37)"`

(C) Felix Zimmermann
