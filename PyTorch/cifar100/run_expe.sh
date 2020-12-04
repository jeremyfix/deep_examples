#!/bin/bash

python3 train.py --batch_size 32 --base_lrate 0.01 
python3 train.py --batch_size 32 --bn --base_lrate 0.01
python3 train.py --batch_size 32 --bn --data_augment --base_lrate 0.01 
python3 train.py --batch_size 32 --bn  --data_augment --l2_reg--dropout --base_lrate 0.01 
