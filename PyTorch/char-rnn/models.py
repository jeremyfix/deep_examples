
# Standard imports
# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, vocab_size, embedding_dim,
                 num_cells, num_layers, num_hidden):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_cells = num_cells
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,
                           num_cells, num_layers,
                          batch_first=True)
        self.classifier = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Linear(num_cells, num_hidden),
            nn.ReLU(),
            # nn.Dropout2d(0.5),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            # nn.Dropout2d(0.5),
            nn.Linear(num_hidden, vocab_size)
        )

    def to_one_hot(self, x):
        # x is a LongTensor (batch, seq_len)
        if len(x.shape) == 1:
            batch, seq_len = 1, x.shape[0]
        else:
            batch, seq_len = x.shape
        device = x.device
        vocab_size = self.vocab_size
        x_idx = x.view(-1, 1)
        inputs = torch.zeros(seq_len*batch, vocab_size, device=device)
        inputs.scatter_(1, x_idx, 1)
        inputs = torch.transpose(inputs.view(batch, seq_len, vocab_size), 0, 1)
        return inputs

    def forward(self, x):
        # x is (batch, seq_len)
        inputs = self.embedding(x)

        output, _ = self.rnn(inputs) # output is (batch, seq, #f)

        output = self.classifier(output)  # (batch, seq_len, vocab_size)

        return output

    def sample(self, x, length):
        with torch.no_grad():
            # x is (seq_len, )
            if len(x.shape) != 1:
                print("""We expect just a one dimensional array for the input when
                      sampling""")
            batch = 1
            device = x.device
            generated_seq = []
            h0 = c0 = torch.zeros(self.num_layers, batch, self.num_cells, device=device)
            next_char_idx = None
            # Feed the network with the starting tensor
            charidxs = np.arange(self.vocab_size)
            for xtidx in x:
                xt = self.embedding(torch.LongTensor([[xtidx]]).to(device))
                output, (h0, c0) = self.rnn(xt, (h0, c0))
                output = self.classifier(output).view(-1)
                # probs may not actually perfectly sum to 1
                probs = F.softmax(output, dim=0).cpu().numpy()
                probs = probs/probs.sum()
                # next_char_idx is used only when leaving the loop
                next_char_idx = np.random.choice(charidxs, p=probs)

            generated_seq.append(next_char_idx)
            for it in range(length):
                xt = self.embedding(torch.LongTensor([[next_char_idx]]).to(device))
                output, (h0, c0) = self.rnn(xt, (h0, c0))
                output = self.classifier(output).view(-1)

                # probs may not actually perfectly sum to 1
                probs = F.softmax(output, dim=0).cpu().numpy()
                probs = probs/probs.sum()
                # next_char_idx is used only when leaving the loop
                next_char_idx = np.random.choice(charidxs, p=probs)
                generated_seq.append(next_char_idx)
            generated_seq = torch.LongTensor(generated_seq).to(device)
            return torch.hstack((x, generated_seq))
