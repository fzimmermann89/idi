CAVE: Hic sunt dracones
    The code is a mess, undocumented and only certain code paths are tested.
    
    


preparation:
--------------
    the simulation needs pycuda and numba (and a bunch of more common stuff)

    those are not installed in any of the psana conda environments, you have to create one for yourself.
    the wrapper assumes it's named "local"
    - quick and dirty - copy my "local" environment)
    - or creating:
    (on pslogin)
        conda create -n local3 python=3.7 numpy mkl ipython ipykernel  jinja2 numba numexpr matplotlib six
        conda activate local3
         pip download pycuda -d ~/tmp    #download the files, won't build without cuda
    (on psanagpu102)
        conda activate local3
        pip install --find-links ~/tmp --no-index pycuda   #build and install
        python -m ipykernel install --user --name local-simulation-env3 --display-name "local simulation(py37)"


    you should now be able to choose this as an kernel in jupyter
    note: do not use pycuda from anaconda. most of the code should work with py3 (untested!)


how to use
-----------

notebooks:
    the simulation can be started from startsim, which uses the psanagpu hosts.
    when the simulation is done, the reconstruction can be started from startrecon (important to wait between the phases),
    plotrecon plots the results.

where to change settings:
    most parameters (sc/fcc/.., lattice constant, number of atoms, fluorescnece energy, detector geometry) can be changed 
    in the startsim notebook where the simulation commands are created.
    the number of photons and the binning can be changed in startrecon.


scripts:
    sim.py - does a simple simulation of the wavefield
    recongrid - reconstruction for different number of photons etc, can be done for small batches 
    recongrid_normalize - final step of the reconstruction after combining the batches


computation time
-----------------
do not simulate more than 1e6 atoms. do not use more than 1e5 photons.