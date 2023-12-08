# from https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py

import os
import sys
import time
import math
import torch
import torch.nn
from torch.nn.modules.module import _addindent
import numpy as np

# https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + " (\n"
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential,
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        total_params += params
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += "  (" + key + "): " + modstr
        if show_weights:
            tmpstr += ", weights={}".format(weights)
        if show_parameters:
            tmpstr += ", parameters={}".format(params)
        tmpstr += "\n"

    tmpstr = tmpstr + ")"
    tmpstr += "\n {} learnable parameters".format(total_params)
    return tmpstr


import torch.utils.data as data


class SplitDataset(data.Dataset):
    """

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, base_dataset, which_to_keep):
        self.base_dataset = base_dataset
        self.which_to_keep = which_to_keep

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.base_dataset[self.which_to_keep[index]]

    def __len__(self):
        return len(self.which_to_keep)

    def __add__(self, other):
        return data.ConcatDataset([self, other])


def split(base_dataset, num_first_set, transform1, transform2):
    """
    Takes a dataset as input
    And splits it into two dataset with num_first_set elements
    in the first dataset
    """
    # We generate a list of indexes for the two datasets
    set1 = base_dataset(transform1)
    set2 = base_dataset(transform2)
    p = np.random.permutation(len(set1))
    first_indexes = list(p[:num_first_set])
    second_indexes = list(p[num_first_set:])
    return SplitDataset(set1, first_indexes), SplitDataset(set2, second_indexes)


def compute_mean_std(loader):
    mean = np.zeros((3, 32, 32), dtype=np.float32)

    total = 0
    # Computing the mean
    for _, (inputs, _) in enumerate(loader, 0):
        npinputs = inputs.numpy()
        mean += inputs.size(0) * np.mean(npinputs, axis=0)
        total += inputs.size(0)
    mean = mean / total

    std = np.zeros((3, 32, 32), dtype=np.float32)
    for _, (inputs, _) in enumerate(loader, 0):
        npinputs = inputs.numpy()
        std += ((npinputs - mean) ** 2).sum(axis=0)

    std = np.sqrt(std / total)
    return torch.from_numpy(mean), torch.from_numpy(std)


class ModelCheckpoint:
    def __init__(self, model, save_path):
        self.model = model
        self.min_loss = None
        self.save_path = save_path

    def update(self, loss):
        if self.min_loss is None or loss < self.min_loss:
            print("+" * 80 + "Saving a better model")
            self.min_loss = loss
            torch.save(self.model.state_dict(), self.save_path)

    @property
    def best(self):
        return self.save_path
