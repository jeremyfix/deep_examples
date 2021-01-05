#!/usr/bin/env python3
"""
Character level RNN, based on an original idea by A. Karpathy
This scripts here learns a character level language model from the 
fables de La Fontaine and texts from other Fabulistes (Fénélon, Bensérade)

python3 main.py train --num_cells 512 --num_layers 2 --slength 30
seems to end up overfitting 
Train metrics :     CE: 0.5426379216358533 | accuracy: 0.8413417754281907
INFO:root:[33/100] Validation:   Loss : 2.206 | Acc : 50.919%
"""

# Standard imports
import argparse
import logging
# External imports
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
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
    embeddings_dim = 64
    num_cells = args.num_cells
    num_layers = args.num_layers
    num_hidden = args.num_hidden
    base_lrate = 0.01
    num_epochs = args.num_epochs
    clip_value = None
    sample_length = 200

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the data
    ds = data.Dataset(args.slength)
    logger.info(f"The vocabulary contains {ds.charmap.vocab_size} elements")

    train_size = int(0.8 * len(ds))
    valid_size = len(ds) - train_size
    train_ds, valid_ds = torch.utils.data.random_split(ds, [train_size, valid_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_ds,
                                               batch_size=batch_size,
                                               shuffle=True)

    # Build the model
    model = models.Model(ds.charmap.vocab_size,
                         embeddings_dim,
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
    # optimizer = optim.SGD(model.parameters(), lr=base_lrate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=10,
                                          gamma=0.5)
    metrics = {'CE': loss, 'accuracy': accuracy}
    start_string = 'LA JUMENT ET '
    generated = sample_from_model(ds.charmap, model, sample_length,
                                  start_string, device)
    logger.info(f"Generated \n>>>\n{generated}\n<<<")

    for i in range(num_epochs):
        train(model, train_loader, loss, optimizer, device, metrics,
              grad_clip=clip_value)
        scheduler.step()
        val_metrics = test(model, valid_loader, device, metrics)
        logger.info("[%d/%d] Validation:   Loss : %.3f | Acc : %.3f%%"% (i,
                                                                   num_epochs,
                                                                   val_metrics['CE'],
                                                                   100.*val_metrics['accuracy']))
        # Sample an example from the model
        generated = sample_from_model(ds.charmap, model, sample_length,
                                      start_string, device)
        logger.info(f"Generated \n>>>\n{generated}\n<<<")

def sample_from_model(charmap, model, length, start_string, device):
    start_input = charmap.encode(start_string)
    start_tensor = torch.LongTensor(start_input)
    start_tensor = start_tensor.to(device)
    # print(f"Start tensor\n{start_tensor}\ncorresponding to\n>>>\n{charmap.decode(start_tensor.view(-1))}\n<<<")
    model.eval()
    generated = model.sample(start_tensor, length)
    return charmap.decode(generated)


def sample(args):
    charmap = data.CharMap.load('charmap')
    start_string = 'Maitre corbeau'
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
                        default=60)
    parser.add_argument("--batch_size", type=int,
                        help="The mini batch size",
                        default=64)
    parser.add_argument("--num_layers", type=int,
                        help="The number of RNN layers",
                        default=2)
    parser.add_argument("--num_cells", type=int,
                        help="The number of cells per RNN layer",
                        default=64)
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
