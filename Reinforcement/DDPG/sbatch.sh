#!/bin/bash
#SBATCH -c 8 # number of cores
#SBATCH --gres=gpu:TitanX:1 # number of gpu requested
#SBATCH -w giptarshish

source ~/tf/bin/activate # activate Tensorflow virtual environment
python sbatch.py

