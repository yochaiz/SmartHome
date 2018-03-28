#!/bin/bash
#SBATCH -c 8 # number of cores
#SBATCH --gres=gpu:1 # number of gpu requested
#SBATCH -w gipbareket

source ~/tf/bin/activate # activate Tensorflow virtual environment
python sbatch.py

