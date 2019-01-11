CAVE: Hic sunt dracones
    The code is a mess, undocumented and only certain code paths are tested.


IDI - INCOHERENT DIFFRACTION IMAGING


content    
------------
ipynb: example notebooks
simulation: simulation of incoherent images
reconstruction: direct and ft based reconstruction

preparation for lcls/psana:
--------------------------
    the package needs pycuda, numba and cython (and a bunch of more common stuff)

    those are not installed in any of the psana conda environments, you have to create one for yourself.

    (on pslogin)
        conda create -n local3 python=3.7 numpy mkl ipython ipykernel cython jinja2 numba numexpr matplotlib six
        conda activate local3
        pip download pycuda -d ~/tmp    #download the files, won't build without cuda
    (on psanagpu102)
        conda activate local3
        pip install --find-links ~/tmp --no-index pycuda   #build and install
        python -m ipykernel install --user --name local-simulation-env3 --display-name "local simulation(py37)"


    you should now be able to choose this as an kernel in jupyter
    note: do not use pycuda from anaconda.

licence
--------
the code will be licenced under BSD 3Clause once working and public. currently there is no oss license.
