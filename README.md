
CAVE: Hic sunt dracones

_The code is a mess, undocumented and only certain code paths are tested._


# IDI - INCOHERENT DIFFRACTION IMAGING

[![https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg](https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg)](https://singularity-hub.org/collections/4824)

content of the repo   
------------
- ipynb: example notebooks
- simulation: simulation of incoherent images
- reconstruction: direct and ft based reconstruction


preparation for slac sdf:
---------------------------
Use Singulariy, if using OOD launcher, use the following to start a jupyterhub

```
    function jupyter() { singularity run --app jupyter --nv -B /sdf,/gpfs,/scratch,/lscratch shub://fzimmermann89/idi $@; }
```


preparation for sacla:
---------------------------
- Download and install miniconda
- `conda create -n local3 python=3.7 numpy mkl mkl-dev ipython ipykernel cython jinja2 numba numexpr matplotlib six scipy`
- `conda activate local3`
- `python -m ipykernel install --user --name local-simulation-env3 --display-name "local simulation(py37)"`


license
--------
the code will be licenced under BSD 3Clause once working and public (and checked if there are any parts under a different license). Currently no license available. 

(C) Felix Zimmermann
