#!/bin/bash
#SBATCH -c 8 # number of cores
#SBATCH --gres=gpu:1 # number of gpu requested
#SBATCH -J "0.9 sequential with rewardScaleFactor & gamma = 0.95"

source ~/tf3/bin/activate # activate Tensorflow virtual environment
python3 sbatch.py

