import os
import shutil
import sys
import glob
import random
import numpy as np

raw_data_path = "raw_data"
dest_data_path = "data"

classes = ["cat", "dog"]

num_samples = 100
perc_valid = 0.1
prob_to_sample=0.1

if not os.path.isdir(raw_data_path):
    print(
        "It seems {} does not exist, did you get the data there ?".format(raw_data_path))
    os.exit(-1)

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
    
    # We shall now move perc_valid * ntrain samples to valid/
    random.shuffle(image_list)
    nvalid = int(perc_valid * ntrain)
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
