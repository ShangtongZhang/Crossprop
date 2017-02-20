#!/bin/bash
#PBS -S /bin/bash

module load python
cd ~/Crossprop
export PYTHONPATH=$PWD
echo "Current working directory is `pwd`"

echo "Starting run at: `date`"
python train@@.py
echo "Job finished with exit code $? at: `date`"