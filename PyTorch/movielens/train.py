import data as mldata
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
import sys
import argparse

    
########## Build the network as :
### Embedding(user)    Embedding(movie)
###           \         /
###                |
###             grade

class Model(nn.Module):
    """Container module with 2 embeddings layers, one dense"""

    def __init__(self, nusers, nmovies, embed_size):
        super(Model, self).__init__()

        self.embed_user = nn.Embedding(nusers, embed_size)
        self.embed_movie = nn.Embedding(nmovies, embed_size)
        self.fc = nn.Linear(embed_size + embed_size)

    def forward(self, input):
        u_emb = self.embed_user(input[0])
        m_emb = self.embed_movie(input[1])
        print(u_emb.shape)
        return 1.5


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.L2loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
100. * batch_idx / len(train_loader), loss.item()))


def main():
    
    parser = argparse.ArgumentParser(description='Movie recommandation with embeddings')
    parser.add_argument('--root_dir', type=str, required=True, action='store')
    parser.add_argument('--no-cuda', action='store_true', default=False,
help='disables CUDA training')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    train_dataset, val_dataset = mldata.train_val_dataset(args.root_dir, 'latest-small', 0.8)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(1, 100):
        train(args, model, device, optimizer, epoch)
    
if __name__ == '__main__':
    main()
