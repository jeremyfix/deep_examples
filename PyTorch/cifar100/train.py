import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

from utils import progress_bar, torch_summarize, split

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

batch_size = 64

#dataset_path = "/home/fix_jer/Datasets"
dataset_path = "/usr/users/ims/fix_jer/Datasets"

classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


# trainset = datasets.CIFAR100(train=True, root=dataset_path, download=True)

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
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),   
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}


    
base_trainset = lambda trans: datasets.CIFAR100(train=True, root=dataset_path, download=True, transform=trans)

trainset, valset = split(base_trainset, 40000, data_transforms['train'], data_transforms['val'])

print("{} samples in the training set".format(len(trainset)))
print("{} samples in the validation set".format(len(valset)))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

testset = datasets.CIFAR100(train=False, root=dataset_path, download=True, transform=data_transforms['test'])
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


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

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0)
    elif 'Linear' in classname:
        nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0)
        
    

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv10 = nn.Conv2d(  3, 32, 3, padding=1)
        self.bn10   = nn.BatchNorm2d(32)
        self.conv11 = nn.Conv2d( 32, 32, 3, padding=1)
        self.bn11   = nn.BatchNorm2d(32)
        
        self.conv20 = nn.Conv2d( 32, 64, 3, padding=1)
        self.bn20   = nn.BatchNorm2d(64)
        self.conv21 = nn.Conv2d( 64, 64, 3, padding=1)
        self.bn21   = nn.BatchNorm2d(64)
        
        self.conv30 = nn.Conv2d( 64,128, 3, padding=1)
        self.bn30   = nn.BatchNorm2d(128)
        self.conv31 = nn.Conv2d(128,128, 3, padding=1)
        self.bn31   = nn.BatchNorm2d(128)
        
        self.conv40 = nn.Conv2d(128,256, 3, padding=1)
        self.bn40   = nn.BatchNorm2d(256)
        self.conv41 = nn.Conv2d(256,256, 3, padding=1)
        self.bn41   = nn.BatchNorm2d(256)
        
        self.fc1   = nn.Linear(256, 500)
        self.drop  = nn.Dropout2d(0.5)
        self.fc2   = nn.Linear(500, 100)

        
    def forward(self, x):
        x = F.elu(self.bn10(self.conv10(x)))
        x = F.elu(self.bn11(self.conv11(x)))
        
        x = F.elu(self.bn20(self.conv20(x)))
        x = F.elu(self.bn21(self.conv21(x)))
        
        x = F.max_pool2d(x, 2)
        
        x = F.elu(self.bn30(self.conv30(x)))
        x = F.elu(self.bn31(self.conv31(x)))
        
        x = F.elu(self.bn40(self.conv40(x)))
        x = F.elu(self.bn41(self.conv41(x)))
        
        x = F.avg_pool2d(x, 16)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.elu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = F.log_softmax(x) # <-- useless if CrossEntropyLoss is used
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
net.apply(weights_init)

#print("{} learnable parameters", len(net.parameters()))
print(torch_summarize(net))

#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(),
                      lr=base_lrate,
                      momentum=0.9,
                      nesterov=True,
                      weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=50,
                                      gamma=0.1)

train_metrics_history = {'times': [], 'loss':[], 'acc':[]}
val_metrics_history = {'times': [], 'loss':[], 'acc':[]}

max_epochs = 200
for epoch in range(max_epochs):  # loop over the dataset multiple times

    train_loss = 0.0
    correct = 0
    total = 0

    # Switch the network to the training mode
    net.train()
    
    scheduler.step()
    for batch_idx, (inputs, targets) in enumerate(trainloader, 0):
		
        # wrap them in Variable
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        inputs, targets = Variable(inputs), Variable(targets)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.data[0]*targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        train_metrics_history['times'].append(epoch + float(total)/(len(trainloader) * batch_size))
        train_metrics_history['acc'].append(correct/float(total))
        train_metrics_history['loss'].append(train_loss/total)
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/total, 100.*correct/total, correct, total))
    
    ##### At the end of an epoch, we compute the metrics on the validation set

    # Switch the network to the testing mode
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valloader, 0):
        # wrap them in Variable
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        inputs, targets = Variable(inputs), Variable(targets)
        # forward + backward + optimize
        outputs = net(inputs)
        
        # print statistics
        val_loss += loss.data[0]*targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    val_loss = val_loss/total
    val_acc = correct/float(total)
    print("[%d/%d] Validation:   Loss : %.3f | Acc : %.3f%%"% (epoch, max_epochs, val_loss, 100.*val_acc))

    val_metrics_history['times'].append(epoch+1)
    val_metrics_history['acc'].append(val_acc)
    val_metrics_history['loss'].append(val_loss)

        
print('Finished Training')


# Switch the network to the testing mode
net.eval()
test_loss = 0.0
correct = 0
total = 0
for batch_idx, (inputs, targets) in enumerate(testloader, 0):
    # wrap them in Variable
    if use_gpu:
        inputs, targets = inputs.cuda(), targets.cuda()
        
    inputs, targets = Variable(inputs), Variable(targets)
    # forward + backward + optimize
    outputs = net(inputs)
    
    # print statistics
    test_loss += loss.data[0]*targets.size(0)
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()
test_acc = 100.*correct/float(total)
test_loss = test_loss/float(total)
print("Test:   Loss : %.3f | Acc : %.3f%%"% (test_loss, test_acc))




plt.figure()

plt.subplot(121)
plt.plot(train_metrics_history['times'], train_metrics_history['acc'])
plt.plot(val_metrics_history['times'], val_metrics_history['acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Num samples')
plt.legend(['train', 'val'], loc='center right')

plt.subplot(122)
plt.plot(train_metrics_history['times'], train_metrics_history['loss'])
plt.plot(val_metrics_history['times'], val_metrics_history['loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Num samples')
plt.legend(['train', 'val'], loc='center right')
plt.suptitle("Test : Loss:%.3f | Acc : %.3f%%" % (test_loss, test_acc))

plt.savefig("expe.pdf", bbox_inches='tight')
