import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

batch_size = 64

classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


# trainset = datasets.CIFAR100(train=True, root="/home/fix_jer/Datasets/", download=True)

# mean = trainset.train_data.mean(axis=0)/255
# std = trainset.train_data.std(axis=0)/255
# print(mean[0,:,:].mean(), mean[1,:,:].mean(), mean[2:,:,:].mean())
# print(std[0,:,:].mean(), std[1,:,:].mean(), std[2:,:,:].mean())

mean = np.array([0.542,0.534,0.474])
std = np.array([0.301,0.295,0.261])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

trainset = datasets.CIFAR100(train=True, root="/home/fix_jer/Datasets/", download=True, transform=data_transforms['train'])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using GPU: {}".torch.cuda.get_device_name())
else:
    print("Using CPU")
    
# # Get a batch of training data
# inputs, classes = next(iter(trainloader))

# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs) # CHW
# outn = out.numpy().transpose(1, 2, 0)*std+mean # HWC

# plt.figure()
# plt.imshow(outn)
# #plt.title(",".join([classnames[i] for i in classes]))
# plt.axis('off')
# plt.show()

################################################# Model

# Architecture
# Transfer function : ELU
# L2 regularization of 0.0005 in all the conv layers
# Convolution blocks : Conv - BN - ELU
# C3s1 x 32, C3s1 x 64, Max2s2, C3s1x128, C3s1x256, GlobalAvg, Dense(500), Dropout(0.5), Dense(100), Softmax

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2D(  3, 32, 3, padding=1)
        self.conv2 = nn.Conv2D( 32, 64, 3, padding=1)
        self.conv3 = nn.Conv2D( 64,128, 3, padding=1)
        self.conv4 = nn.Conv2D(128,256, 3, padding=1)
        self.fc1   = nn.Linear( 64, 500)
        self.fc2   = nn.Linear(500, 100)

        
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.conv2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Base lrate : 0.01
