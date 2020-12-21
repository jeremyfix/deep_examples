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

    model = models.Model(ds.charmap.vocab_size,
                         num_cells,
                         num_layers,
                         num_hidden)

    X, y = next(iter(train_loader))
    model(X)

    # Build up the loss/optimizer/..


def sample(args):
    
    charmap = data.CharMap.load('charmap')
    start_string = charmap.start_line + 'Maitre corbeau'
    start_input = charmap.encode(start_string)
    # Build up the warm up tensor and add the batch size dimension
    start_tensor = torch.Tensor(start_input).long().unsqueeze(dim=0)
    print(f"Start tensor\n{start_tensor}\ncorresponding to\n>>>\n{charmap.decode(start_tensor.view(-1))}\n<<<")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        choices=['train', 'sample'])
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
