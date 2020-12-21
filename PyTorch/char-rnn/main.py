#!/usr/bin/env python3
"""
Character level RNN, based on an original idea by A. Karpathy
This scripts here learns a character level language model from the 
fables de La Fontaine.
"""

# Standard imports
import argparse
# External imports
import torch
# Local imports
import data
import models

def train(args):

    batch_size = args.batch_size
    num_cells = args.num_cells
    num_layers = args.num_layers
    num_hidden = args.num_hidden

    # Load the data
    ds = data.Dataset(args.slength)

    train_loader = torch.utils.data.DataLoader(dataset=ds,
                                               batch_size=batch_size,
                                               shuffle=False)

    model = models.Model(ds.vocab_size, num_cells, num_layers,
                         num_hidden)

    X, y = next(iter(train_loader))
    print(X.shape)
    model(X)


def sample(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        choices=['train', 'predict'])
    parser.add_argument("--slength", type=int,
                        help="The length of the strings for training",
                        default=30)
    parser.add_argument("--batch_size", type=int,
                        help="The mini batch size",
                        default=32)
    parser.add_argument("--num_layers", type=int,
                        help="The number of RNN layers",
                        default=1)
    parser.add_argument("--num_cells", type=int,
                        help="The number of cells per RNN layer",
                        default=128)
    parser.add_argument("--num_hidden", type=int,
                        help="The number of hidden units for the dense output",
                        default=128)

    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    else:
        sample(args)
