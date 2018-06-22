# If segfault,
# export LD_LIBRARY_PATH=/usr/lib/nvidia-384

# See ?
# https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf

import data as mldata
import model as rs_model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import sys
import argparse

use_tensorboard = True
try:
    from tensorboardXy import SummaryWriter
except:
    use_tensorboard = False

use_torch_summary = True
try:
    import torchsummary
except:
    use_torch_summary = False

def print_log(epoch, train_loss, progress):
    """
    epoch : int
    train_acc : float
    train_loss : float
    progress in [0, 1] : progression of the epoch
    """
    length_bar = 40
    prog_str = ("#"*math.floor(length_bar * progress)) + ("-"*math.floor(length_bar*(1.0-progress)))
    sys.stdout.write('\rEpoch {}: [{}] Train loss : {:.6f}'.format(epoch, prog_str, train_loss))
    sys.stdout.flush()
    
def train(model, device, train_loader, val_loader, optimizer, epoch, writer, best_val):
    model.train()
    loss = nn.MSELoss(size_average=False)
    num_train_samples = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        predicted = model(data)

        output = loss(predicted, target)
        output.backward()
        optimizer.step()
        num_train_samples += len(data)
        train_loss += output.item()
        if batch_idx % 100 == 0:
            print_log(epoch, train_loss/num_train_samples, batch_idx / len(train_loader))
        if writer:
            writer.add_scalar('data/train_loss', output.item(), epoch + batch_idx / len(train_loader))
    print()

    model.eval()
    val_loss = 0.0
    loss = nn.MSELoss(size_average=False)
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        predicted = model(data)
        output = loss(predicted, target)
        val_loss += output.item()
    val_loss /= len(val_loader.dataset)
    if best_val is None or best_val > val_loss:
        model.save("best_model.pt")
        best_val = val_loss
        print("Better model saved")
    if writer:
        writer.add_scalar('data/val_loss', val_loss, epoch+1)
    print('Validation loss : {:.6f}'.format(val_loss))
    return best_val
    
def main():

    print("parsing")
    parser = argparse.ArgumentParser(description='Movie recommandation with embeddings')
    parser.add_argument('--root_dir', type=str, required=True, action='store')
    parser.add_argument('--no-cuda', action='store_true', default=False,
help='disables CUDA training')
    parser.add_argument('--embed_size', type=int, required=True, action='store')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    embed_size = args.embed_size
    
    print("dset")
    train_dataset, val_dataset, nusers, nmovies, rating_range = mldata.train_val_dataset(args.root_dir, '20m', 0.8)
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("loading data")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    writer = None
    if use_tensorboard:
        writer = SummaryWriter()
        

    model = rs_model.Model(nusers, nmovies, embed_size, rating_range).to(device)
    #if use_torch_summary:
    #    torchsummary.summary(model, (2, 2))
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    best_val = None
    for epoch in range(1, 25):
        best_val = train(model, device, train_loader, val_loader, optimizer, epoch, writer, best_val)
    
if __name__ == '__main__':
    main()
