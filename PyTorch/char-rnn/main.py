#!/usr/bin/env python3
"""
Character level RNN, based on an original idea by A. Karpathy
This scripts here learns a character level language model from the 
fables de La Fontaine.
"""

# Standard imports
import argparse
import logging
# External imports
import torch
import torch.optim as optim
import deepcs
from deepcs.training import train
from deepcs.testing import test
from deepcs.metrics import accuracy
import deepcs.display
# Local imports
import data
import models

def trainnet(args):
    logger = logging.getLogger()

    batch_size = args.batch_size
    num_cells = args.num_cells
    num_layers = args.num_layers
    num_hidden = args.num_hidden
    base_lrate = 0.01
    num_epochs = args.num_epochs
    clip_value = 5
    sample_length = 100

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the data
    ds = data.Dataset(args.slength)
    logger.info(f"The vocabulary contains {ds.charmap.vocab_size} elements")

    train_loader = torch.utils.data.DataLoader(dataset=ds,
                                               batch_size=batch_size,
                                               shuffle=True)

    # Build the model
    model = models.Model(ds.charmap.vocab_size,
                         num_cells,
                         num_layers,
                         num_hidden)
    print(deepcs.display.torch_summarize(model))
    model.to(device)

    # Build up the loss/optimizer/metrics
    def loss(seq_outputs, seq_targets):
        # seq_outputs is (batch, seq_len, vocab_size)
        # set_targets is (batch, seq_len)
        batch, seq_len = seq_targets.shape
        seq_outputs = seq_outputs.view(batch*seq_len, -1)
        seq_targets = seq_targets.view(-1)
        return torch.nn.CrossEntropyLoss()(seq_outputs, seq_targets)

    # loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lrate)

    metrics = {'CE': loss, 'accuracy': accuracy}
    start_string = ds.charmap.start_line + 'Maitre corbeau'

    for i in range(num_epochs):
        train(model, train_loader, loss, optimizer, device, metrics,
              grad_clip=clip_value)
        # Sample an example from the model
        generated = sample_from_model(ds.charmap, model, sample_length,
                                      start_string, device)
        logger.info(f"Generated \n>>>\n{generated}\n<<<")

def sample_from_model(charmap, model, length, start_string, device):
    start_input = charmap.encode(start_string)
    start_tensor = torch.LongTensor(start_input, device=device)
    # print(f"Start tensor\n{start_tensor}\ncorresponding to\n>>>\n{charmap.decode(start_tensor.view(-1))}\n<<<")
    model.eval()
    generated = model.sample(start_tensor, length)
    return charmap.decode(generated)


def sample(args):
    charmap = data.CharMap.load('charmap')
    start_string = charmap.start_line + 'Maitre corbeau'
    model = None #TODO
    device = None
    raise NotImplementedError
    sample_from_model(charmap, model, 50, start_string, device)


if __name__ == '__main__':

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

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
                        default=123)
    parser.add_argument("--num_hidden", type=int,
                        help="The number of hidden units for the dense output",
                        default=128)
    parser.add_argument("--num_epochs", type=int,
                        help="The number of epochs for training",
                        default=100)

    args = parser.parse_args()
    if args.command == 'train':
        trainnet(args)
    else:
        sample(args)
