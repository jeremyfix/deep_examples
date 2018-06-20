# If segfault,
# export LD_LIBRARY_PATH=/usr/lib/nvidia-384

import data as mldata
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import argparse
#from tensorboardX import SummaryWriter
    
########## Build the network as :
### Embedding(user)    Embedding(movie)
###           \         /
###                |
###             grade

class Model(nn.Module):
    """Container module with 2 embeddings layers, one dense"""

    def __init__(self, nusers, nmovies, embed_size, rating_range):
        super(Model, self).__init__()

        self.rating_range = rating_range
        self.embed_user = nn.Embedding(nusers, embed_size)
        self.embed_user.weight.data.normal_(0, 0.01)
        self.bias_user = nn.Embedding(nusers, 1)
        self.embed_movie = nn.Embedding(nmovies, embed_size)
        self.embed_movie.weight.data.normal_(0, 0.01)
        self.bias_movie = nn.Embedding(nmovies, 1)

    def forward(self, inp):
        u_emb = self.embed_user(inp[:,0])
        u_b = self.bias_user(inp[:,0])
        m_emb = self.embed_movie(inp[:,1])
        m_b = self.bias_movie(inp[:,1])
        y_pred = (u_emb * m_emb).sum(1) + u_b.squeeze() + m_b.squeeze()
        y_pred = F.sigmoid(y_pred) * (self.rating_range[1] - self.rating_range[0]) + self.rating_range[0]
        return y_pred.view(y_pred.size()[0])


def train(model, device, train_loader, val_loader, optimizer, epoch):#, writer):
    model.train()
    loss = nn.MSELoss()
    num_train_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        predicted = model(data)

        output = loss(predicted, target)
        output.backward()
        optimizer.step()
        num_train_samples += len(data)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, num_train_samples, len(train_loader.dataset),
100. * batch_idx / len(train_loader), output.item()))
#writer.add_scalar('data/train_loss', output.item(), epoch + batch_idx / len(train_loader))
            
    val_loss = 0.0
    loss = nn.MSELoss(size_average=False)
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        predicted = model(data)
        output = loss(predicted, target)
        val_loss += output.item()
    val_loss /= len(val_loader.dataset)
#writer.add_scalar('data/val_loss', val_loss, epoch+1)
    print('Validation loss : {:.6f}'.format(val_loss))
    
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
    train_dataset, val_dataset, nusers, nmovies, rating_range = mldata.train_val_dataset(args.root_dir, 'latest-small', 0.8)
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("loading data")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


#writer = SummaryWriter()
    

    print("moving model")
    model = Model(nusers, nmovies, embed_size, rating_range).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print("training")
    for epoch in range(1, 100):
        train(model, device, train_loader, val_loader, optimizer, epoch)#, writer)
    
if __name__ == '__main__':
    main()
