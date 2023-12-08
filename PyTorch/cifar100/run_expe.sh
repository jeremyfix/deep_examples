#!/bin/bash

#SBATCH --job-name=cifar100
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00

python3 -m venv $TMPDIR/venv
source $TMPDIR/venv/bin/activate
python -m pip install torch torchvision deepcs matplotlib numpy

python train.py --batch_size 32 --base_lrate 0.01 
python train.py --batch_size 32 --base_lrate 0.01 --bn
python train.py --batch_size 32 --base_lrate 0.01 --bn --data_augment 
python train.py --batch_size 32 --base_lrate 0.01 --bn --data_augment --l2_reg --dropout --lsmooth
