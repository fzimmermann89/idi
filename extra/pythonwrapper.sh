#!/bin/bash

#set path for cuda
export PATH=/usr/local/cuda/bin:$PATH
#export NUMBA_NUM_THREADS=12
#activate environment
source /reg/g/psdm/etc/psconda.sh 
conda activate local3

#start job
echo started on $(hostname)..
python "$@"


