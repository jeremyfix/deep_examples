import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from momentum_sgd import TestSGD
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

# pip3 install nibabel tqdm
# pip3 install git+https://github.com/ncullen93/torchsample
import torchsample
from torchsample.transforms import RandomAffine, RandomFlip, RandomTranslate

from utils import progress_bar, torch_summarize, compute_mean_std

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_augment',
    help='Specify if you want to use data augmentation',
    action='store_true'
)

parser.add_argument(
    '--dropout',
    help='Specify to use dropout',
    action='store_true'
)

parser.add_argument(
    '--l2_reg',
    help='Specify to use l2_reg',
    action='store_true'
)

parser.add_argument(
    '--bn',
    help='Specify to use batch normalization',
    action='store_true'
)

parser.add_argument(
    '--base_lrate',
    help='Which base learning rate to use',
    type=float,
    default=0.001
)

parser.add_argument(
    '--batch_size',
    required=True,
    type=int
)

args = parser.parse_args()

use_dataset_augmentation = args.data_augment
use_dropout = args.dropout
use_bn = args.bn
base_lrate = args.base_lrate
batch_size = args.batch_size
use_l2_reg = args.l2_reg
valid_size = 0.2
num_workers = 2

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using GPU{}".format(torch.cuda.current_device()))
else:
    print("Using CPU")

pin_memory = use_gpu

#dataset_path = "/home/fix_jer/Datasets"
#dataset_path = "/usr/users/ims/fix_jer/Datasets"
dataset_path = "/opt/Datasets/"

classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

####### Computing mean/std images from the training set
####### and splitting into train/val
train_dataset = datasets.CIFAR100(train=True, root=dataset_path, download=True, transform=transforms.ToTensor())

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=2, pin_memory=pin_memory)


#mean_tensor = torch.Tensor(np.array([0.5,0.5,0.5], dtype=np.float32))
#std_tensor = torch.Tensor(np.array([0.3,0.3,0.3], dtype=np.float32))

mean_tensor, std_tensor = compute_mean_std(train_loader)



data_transforms = {
    'train': None,
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x.sub(mean_tensor)).div(std_tensor))
    ]),   
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x.sub(mean_tensor)).div(std_tensor))
    ])
}

if use_dataset_augmentation:
    data_transforms['train'] = transforms.Compose([
        transforms.ToTensor(),
        RandomTranslate((5./32., 5./32.), interp='nearest'),
        RandomAffine(zoom_range=(0.8, 1.2)),
        RandomFlip(h=True, v=False),
        transforms.Lambda(lambda x: (x.sub(mean_tensor)).div(std_tensor))
    ])
else:
    data_transforms['train'] = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x.sub(mean_tensor)).div(std_tensor))
    ])

train_dataset = datasets.CIFAR100(train=True, root=dataset_path, download=True, transform=data_transforms['train'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)

valid_dataset = datasets.CIFAR100(train=True, root=dataset_path, download=True, transform=data_transforms['valid'])
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, pin_memory=pin_memory)


print("{} samples in the training set".format(len(train_idx)))
print("{} samples in the validation set".format(len(valid_idx)))


test_dataset = datasets.CIFAR100(train=False, root=dataset_path, download=True, transform=data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)



    
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

    def __init__(self, use_dropout, use_batchnorm, use_l2reg):
        super(Net, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.use_l2reg = use_l2reg
        
        batchnorm_momentum = 0.99
        batchnorm_epsilon = 1e-3
        
        self.conv10 = nn.Conv2d(  3, 32, 3, padding=1)
        if self.use_batchnorm:
            self.bn10   = nn.BatchNorm2d(32, eps=batchnorm_epsilon, momentum=batchnorm_momentum)
        self.elu10  = nn.ELU(inplace=True)
        self.conv11 = nn.Conv2d( 32, 32, 3, padding=1)
        if self.use_batchnorm:
            self.bn11   = nn.BatchNorm2d(32, eps=batchnorm_epsilon, momentum=batchnorm_momentum)
        self.elu11  = nn.ELU(inplace=True)
        
        self.conv20 = nn.Conv2d( 32, 64, 3, padding=1)
        if self.use_batchnorm:
            self.bn20   = nn.BatchNorm2d(64, eps=batchnorm_epsilon, momentum=batchnorm_momentum)
        self.elu20  = nn.ELU(inplace=True)
        self.conv21 = nn.Conv2d( 64, 64, 3, padding=1)
        if self.use_batchnorm:
            self.bn21   = nn.BatchNorm2d(64, eps=batchnorm_epsilon, momentum=batchnorm_momentum)
        self.elu21  = nn.ELU(inplace=True)
        
        self.conv30 = nn.Conv2d( 64,128, 3, padding=1)
        if self.use_batchnorm:
            self.bn30   = nn.BatchNorm2d(128, eps=batchnorm_epsilon, momentum=batchnorm_momentum)
        self.elu30  = nn.ELU(inplace=True)
        self.conv31 = nn.Conv2d(128,128, 3, padding=1)
        if self.use_batchnorm:
            self.bn31   = nn.BatchNorm2d(128, eps=batchnorm_epsilon, momentum=batchnorm_momentum)
        self.elu31  = nn.ELU(inplace=True)
        
        self.conv40 = nn.Conv2d(128,256, 3, padding=1)
        if self.use_batchnorm:
            self.bn40   = nn.BatchNorm2d(256, eps=batchnorm_epsilon, momentum=batchnorm_momentum)
        self.elu40  = nn.ELU(inplace=True)
        self.conv41 = nn.Conv2d(256,256, 3, padding=1)
        if self.use_batchnorm:
            self.bn41   = nn.BatchNorm2d(256, eps=batchnorm_epsilon, momentum=batchnorm_momentum)
        self.elu41  = nn.ELU(inplace=True)
        
        self.fc1   = nn.Linear(256, 500)
        self.elu1  = nn.ELU(inplace=True)
        if self.use_dropout:
            self.drop  = nn.Dropout2d(0.5)
        self.fc2   = nn.Linear(500, 100)

        self.loss = nn.NLLLoss()
        if use_l2reg:
            pass
        
        self.init()
        
    def init(self):
        for m in self.modules():
            if   isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv10(x)
        if self.use_batchnorm:
            x = self.bn10(x)
        x = self.elu10(x)

        x = self.conv11(x)
        if self.use_batchnorm:
            x = self.bn11(x)
        x = self.elu11(x)
        
        #x = F.max_pool2d(x, 2)

        x = self.conv20(x)
        if self.use_batchnorm:
            x = self.bn20(x)
        x = self.elu20(x)

        x = self.conv21(x)
        if self.use_batchnorm:
            x = self.bn21(x)
        x = self.elu21(x)
        
        x = F.max_pool2d(x, 2)

        x = self.conv30(x)
        if self.use_batchnorm:
            x = self.bn30(x)
        x = self.elu30(x)

        x = self.conv31(x)
        if self.use_batchnorm:
            x = self.bn31(x)
        x = self.elu31(x)
        
        #x = F.max_pool2d(x, 2)

        x = self.conv40(x)
        if self.use_batchnorm:
            x = self.bn40(x)
        x = self.elu40(x)

        x = self.conv41(x)
        if self.use_batchnorm:
            x = self.bn41(x)
        x = self.elu41(x)

        x = F.avg_pool2d(x, x.size()[-1])
        
        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        x = self.elu1(x)

        if self.use_dropout:
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


    
net = Net(use_dropout, use_bn, use_l2_reg)
if(use_gpu):
    net.cuda()

#print("{} learnable parameters", len(net.parameters()))
print(torch_summarize(net))

#criterion = nn.CrossEntropyLoss()
criterion = net.loss
#optimizer = optim.SGD(net.parameters(),
#                      lr=base_lrate,
#                      momentum=0.9)
optimizer = TestSGD(net.parameters(),
                      lr=base_lrate,
                      momentum=0.9)

scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=50,
                                      gamma=0.1)

train_metrics_history = {'times': [], 'loss':[], 'acc':[]}
val_metrics_history = {'times': [], 'loss':[], 'acc':[]}

max_epochs = 150
for epoch in range(max_epochs):  # loop over the dataset multiple times

    train_loss = 0.0
    correct = 0
    total = 0

    # Switch the network to the training mode
    net.train()
    
    scheduler.step()
    for batch_idx, (inputs, targets) in enumerate(train_loader, 0):
		
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
        correct += torch.sum(predicted == targets.data)
        
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/total, 100.*correct/total, correct, total))

    train_loss = train_loss / total
    train_acc = correct / total
    train_metrics_history['times'].append(epoch + 1)
    train_metrics_history['acc'].append(train_acc)
    train_metrics_history['loss'].append(train_loss)
        
    ##### At the end of an epoch, we compute the metrics on the validation set

    
    # Switch the network to the testing mode
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valid_loader, 0):
        # wrap them in Variable
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        inputs, targets = Variable(inputs), Variable(targets)
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        # print statistics
        val_loss += loss.data[0]*targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += torch.sum(predicted == targets.data)
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
for batch_idx, (inputs, targets) in enumerate(test_loader, 0):
    # wrap them in Variable
    if use_gpu:
        inputs, targets = inputs.cuda(), targets.cuda()
        
    inputs, targets = Variable(inputs), Variable(targets)
    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    
    # print statistics
    test_loss += loss.data[0]*targets.size(0)
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += torch.sum(predicted == targets.data)
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


suptitle = "Test : Loss:%.3f | Acc : %.3f%%;" % (test_loss, test_acc)
pdf_filename = ""
if use_dataset_augmentation:
    suptitle += " dataAugment "
    pdf_filename += "dataAugment"
else:
    suptitle += " - "
if use_dropout:
    suptitle += " dropout "
    pdf_filename += "_dropout_"
else:
    suptitle += " - "
if use_l2_reg:
    suptitle += " l2 "
    pdf_filename += "_l2_"
else:
    suptitle += " - "
if use_bn:
    suptitle += " BN "
    pdf_filename += "_BN_"
else:
    suptitle += " - "
suptitle += " lr{} ".format(base_lrate)
suptitle += " bs{} ".format(batch_size)
pdf_filename += "_lr{}_".format(base_lrate)
pdf_filename += "_bs{}_".format(batch_size)


plt.suptitle(suptitle)

plt.savefig(pdf_filename+".pdf", bbox_inches='tight')
