#!/bin/bash
#SBATCH -c 8 # number of cores
#SBATCH --gres=gpu:TitanX:1 # number of gpu requested
#SBATCH -w giptarshish
#SBATCH -J "step 13 experiments"

source ~/tf3/bin/activate # activate Tensorflow virtual environment
python3 sbatch.py

