#!/bin/bash

python3 train.py --batch_size 32 --base_lrate 0.01 
python3 train.py --batch_size 32 --base_lrate 0.01 --bn
python3 train.py --batch_size 32 --base_lrate 0.01 --bn --data_augment 
python3 train.py --batch_size 32 --base_lrate 0.01 --bn --data_augment --l2_reg --dropout --lsmooth
