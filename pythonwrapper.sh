#!/bin/bash
# Wrapper for use at SLAC. Usage: ./pythonwrapper nameofenvironment scriptname arguments

envname="$1"; shift
HOST=`hostname`

#load env
if [ -e /reg/g/psdm/etc/psconda.sh ]; then
#psana
    export PATH=/usr/local/cuda/bin:$PATH
    source /reg/g/psdm/etc/psconda.sh 
    conda activate $envname

elif [[ $HOST == *"cent7"* ]] || [[ $HOST == *"cryo"* ]]; then
#centos or slacgpu
    echo cent
    export PATH=~/local/bin:$PATH
    export LD_LIBRARY_PATH=~/local/lib:$LD_LIBRARY_PATH
    export MODULEPATH=/usr/share/Modules/modulefiles:/etc/modulefiles:/afs/slac/package/spack/share/spack/modules/linux-centos7-x86_64
    source /usr/share/Modules/init/sh
    module load py-virtualenv-15.1.0-gcc-4.8.5-noxuoys cuda-9.1.85-gcc-4.9.4-fd4g2ts miniconda2-4.5.4-gcc-4.8.5-z77dhmy
    source /afs/slac.stanford.edu/package/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/miniconda2-4.5.4-z77dhmywrtswwoebkrd7wa5rkpn42jmz/etc/profile.d/conda.sh
    export _CONDA_EXE=~/local/bin/conda
    export CONDA_PYTHON_EXE=/afs/slac.stanford.edu/package/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/miniconda2-4.5.4-z77dhmywrtswwoebkrd7wa5rkpn42jmz/bin/python
    conda activate $envname
    aklog
else
    echo dont know how to load $envname on $HOST
fi

#start job
echo started on $(hostname)..
python "$@"

