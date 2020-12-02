import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import argparse
import deepcs
from deepcs.metrics import accuracy
from deepcs.training import train
from deepcs.testing import test

# Local imports
import models
from utils import compute_mean_std


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
    device = torch.device('gpu')
else:
    print("Using CPU")
    device = torch.device('cpu')

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
train_idx, valid_idx = indices[-128:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=2, pin_memory=pin_memory)

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

model = models.Net(use_dropout, use_bn, use_l2_reg)
model = model.to(device)

# Display information about the model
summary_text = "Summary of the model architecture\n"+ \
        "=================================\n" + \
        f"{deepcs.display.torch_summarize(model)}\n"

print(summary_text)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=base_lrate,
                      momentum=0.9)
metrics = {
    'CE': loss,
    'accuracy': accuracy
}
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=50,
                                      gamma=0.5)

train_metrics_history = {'times': [], 'loss':[], 'acc':[]}
val_metrics_history = {'times': [], 'loss':[], 'acc':[]}

max_epochs = 150
for epoch in range(max_epochs):  # loop over the dataset multiple times

    train(model,
          train_loader,
          loss,
          optimizer,
          device,
          metrics
         )
    scheduler.step()

    train_metrics = test(model, train_loader, device, metrics)
    train_metrics_history['times'].append(epoch + 1)
    train_metrics_history['acc'].append(train_metrics['accuracy'])
    train_metrics_history['loss'].append(train_metrics['CE'])

    ##### At the end of an epoch, we compute the metrics on the validation set
    val_metrics = test(model, valid_loader, device, metrics)
    print("[%d/%d] Validation:   Loss : %.3f | Acc : %.3f%%"% (epoch,
                                                               max_epochs,
                                                               val_metrics['CE'],
                                                               100.*val_metrics['accuracy']))

    val_metrics_history['times'].append(epoch+1)
    val_metrics_history['acc'].append(val_metrics['accuracy'])
    val_metrics_history['loss'].append(val_metrics['CE'])


print('Finished Training')

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

plt.savefig(pdf_filename+"_sched.pdf", bbox_inches='tight')
