#!/bin/bash
#PBS -S /bin/bash

module load python
cd ~/PaperReplication/Crossprop
export PYTHONPATH=$PWD
echo "Current working directory is `pwd`"

echo "Starting run at: `date`"
python train@@.py
echo "Job finished with exit code $? at: `date`"