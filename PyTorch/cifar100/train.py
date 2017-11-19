import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
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


# trainset = datasets.CIFAR100(train=True, root="/usr/users/ims/fix_jer/Datasets/", download=True)

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

trainset = datasets.CIFAR100(train=True, root="/usr/users/ims/fix_jer/Datasets/", download=True, transform=data_transforms['train'])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using GPU{}".format(torch.cuda.current_device()))
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
        self.conv1 = nn.Conv2d(  3, 32, 3, padding=1)
        nn.init.xavier_uniform(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d( 32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d( 64,128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        self.fc1   = nn.Linear(256, 500)
        self.drop  = nn.Dropout2d(0.5)
        self.fc2   = nn.Linear(500, 100)

        
    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.bn4(self.conv4(x)))
        x = F.avg_pool2d(x, 16)
        x = x.view(-1, self.num_flat_features(x))
        x = F.elu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
		# x = F.softmax(x) <-- useless if CrossEntropyLoss is used
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Base lrate : 0.01

base_lrate = 0.01

net = Net()
if(use_gpu):
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                      lr=base_lrate,
                      momentum=0.9,
                      nesterov=True)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=50,
                                      gamma=0.1)

for epoch in range(200):  # loop over the dataset multiple times

    running_loss = 0.0
    scheduler.step()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
		
        # wrap them in Variable
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        inputs = Variable(inputs)
        labels = Variable(labels)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
