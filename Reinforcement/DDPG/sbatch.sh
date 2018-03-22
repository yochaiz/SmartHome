#!/bin/bash
#SBATCH -c 8 # number of cores
#SBATCH --gres=gpu:1 # number of gpu requested
#SBATCH -w giptarshish

source ~/tf/bin/activate # activate Tensorflow virtual environment
echo "test"
python train.py 0 --random --k 32 --desc "fixed future action (discrete) + knn over all possible discrete actions" &
echo "1 is running, going to sleep"
sleep 5 # sleep for few seconds
echo "good morning, running 2"
python train.py 0 --random --desc "fixed future action (discrete) + default knn size"
echo "2 is running"
