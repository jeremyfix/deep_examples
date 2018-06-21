import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Container module with 2 embeddings layers, one dense"""

    def __init__(self, nusers, nmovies, embed_size, ratings_range):
        super(Model, self).__init__()

        self.nusers = nusers
        self.nmovies = nmovies
        self.embed_size = embed_size
        self.ratings_range = ratings_range
        
        self.drop = nn.Dropout(0.2, inplace=True)
        self.embed_user = nn.Embedding(nusers, embed_size)
        self.embed_user.weight.data.normal_(0, 0.01)
        self.bias_user = nn.Embedding(nusers, 1)
        self.embed_movie = nn.Embedding(nmovies, embed_size)
        self.embed_movie.weight.data.normal_(0, 0.01)
        self.bias_movie = nn.Embedding(nmovies, 1)

    def forward0(self, inp):
        inp = inp.long()
        u_emb = self.embed_user(inp[:,0])
        u_b = self.bias_user(inp[:,0])
        m_emb = self.embed_movie(inp[:,1])
        m_b = self.bias_movie(inp[:,1])
        y_pred = (u_emb * m_emb).sum(1) + u_b.squeeze() + m_b.squeeze()
        y_pred = F.sigmoid(y_pred) * (self.ratings_range[1] - self.ratings_range[0]) + self.ratings_range[0]
        return y_pred.view(y_pred.size()[0])
    
    def forward(self, inp):
        inp = inp.long()
        u_emb = self.embed_user(inp[:,0])
        self.drop(u_emb)
        u_b = self.bias_user(inp[:,0])
        m_emb = self.embed_movie(inp[:,1])
        self.drop(m_emb)
        m_b = self.bias_movie(inp[:,1])
        y_pred = (u_emb * m_emb).sum(1) + u_b.squeeze() + m_b.squeeze()
        y_pred = F.sigmoid(y_pred) * (self.ratings_range[1] - self.ratings_range[0]) + self.ratings_range[0]
        return y_pred.view(y_pred.size()[0])

    def save(self, filename):
        state = { "nusers": self.nusers,
                  "nmovies": self.nmovies,
                  "embed_size": self.embed_size,
                  "ratings_range": self.ratings_range,
                  "state_dict": self.state_dict()}
        torch.save(state, filename)
