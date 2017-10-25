import os
import shutil
import sys
import glob
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--sourcedir',
    type=str,
    default="/opt/DeepLearning/Datasets/CatsDogs",
    help='The directory where the raw_data are stored'
)
parser.add_argument(
    '--destdir',
    type=str,
    default="./data",
    help='The directory where the split data will be stored'
)
parser.add_argument(
    '--probvalid',
    type=float,
    default=0.1,
    help='The probability to put a sample in the validation set. In mean, we will have numsamples * set_size  samples in the validation set'
)
parser.add_argument(
    '--probsample',
    type=float,
    default=0.1,
    help='The probability to put a sample in the small dataset. In mean, we will have numsamples * set_size  samples in the small dataset. This small dataset is for debugging purpose.'
)

args = parser.parse_args()

raw_data_path = os.path.expanduser(args.sourcedir)
dest_data_path = args.destdir

classes = ["cat", "dog"]

prob_to_valid = args.probvalid
prob_to_sample= args.probsample

if not os.path.isdir(raw_data_path):
    print(
        "It seems {} does not exist, did you get the data there ?".format(raw_data_path))
    sys.exit(-1)

def question(str):
    while True:
        answer = input(str)
        if(answer is "y"):
            return True
        elif answer is "n":
            return False

if os.path.isdir(dest_data_path):
    answer = question("{} already exists, should I remove it ? [y/n]".format(dest_data_path))
    if answer:
        print("Removing {}".format(dest_data_path))
        shutil.rmtree(dest_data_path)
    else:
        print("Aborting")
        sys.exit(0)

os.mkdir(dest_data_path)

print("Copying the test data")
shutil.copytree("{}/test".format(raw_data_path), "{}/test/images".format(dest_data_path))

print("Splitting the data into data/train, data/val")

for c in classes:
    path=[dest_data_path, "train",c]
    os.makedirs(os.path.join(*path))
    path = [raw_data_path,"train","{}*.jpg".format(c)]
    ntrain=0
    image_list=[]
    for f in glob.glob(os.path.join(*path)):
        pathdest = [dest_data_path, "train", c]
        fdest = os.path.basename(f)[len(c)+1:]
        fdest = os.path.join(*(pathdest + [fdest]))
        image_list.append(fdest)
        shutil.copy(f, fdest)
        ntrain += 1
    
    # We shall now move prob_to_valid * ntrain samples to valid/
    random.shuffle(image_list)
    nvalid = int(prob_to_valid * ntrain)
    path=[dest_data_path, "valid",c]
    os.makedirs(os.path.join(*path))
    for f in image_list[:nvalid]:
        os.rename(f, os.path.join(*(path + [os.path.basename(f)])))
    
    
print("Copying few elements into sample/train, sample/val")
for c in classes:
    sample_path=[dest_data_path, "sample","train", c]
    os.makedirs(os.path.join(*sample_path))
    data_path=[dest_data_path , "train", c, "*.jpg"]
    for f in glob.glob(os.path.join(*data_path)):
        if(random.random() < prob_to_sample):
            shutil.copyfile(f, os.path.join(*(sample_path+[os.path.basename(f)])))
        
    sample_path=[dest_data_path, "sample","valid", c]
    os.makedirs(os.path.join(*sample_path))
    data_path=[dest_data_path , "valid", c, "*.jpg"]
    for f in glob.glob(os.path.join(*data_path)):
        if(random.random() < prob_to_sample):
            shutil.copyfile(f, os.path.join(*(sample_path+[os.path.basename(f)])))    
