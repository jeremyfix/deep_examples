
# Standard imports
# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, vocab_size, num_embeddings,
                 num_cells, num_layers, num_hidden):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.num_embeddings = num_embeddings
        self.num_cells = num_cells
        self.num_layers = num_layers
        self.embedding = nn.Linear(vocab_size, num_embeddings)
        self.rnn = nn.LSTM(num_embeddings, num_cells, num_layers)
        self.classifier = nn.Sequential(
            # nn.Dropout2d(0.5),
            # nn.Linear(num_cells, num_hidden),
            # nn.ReLU(),
            # nn.Dropout2d(0.5),
            # nn.Linear(num_hidden, num_hidden),
            # nn.ReLU(),
            # nn.Dropout2d(0.5),
            nn.Linear(num_cells, vocab_size)
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
        (batch, seq_len), vocab_size = x.shape, self.vocab_size

        device = x.device
        # the inputs to the LSTM must be (seq_len, batch, input_size)
        # we first need to one-hot encode the inputs
        inputs = self.to_one_hot(x)

        inputs = self.embedding(inputs.reshape(seq_len*batch, -1)).reshape(seq_len, batch, self.num_embeddings)

        # input to the LSTM must be (seq_len, batch, vocab_size)
        # h0, c0 are (num_layers, batch, hidden_size)
        h0 = c0 = torch.zeros(self.num_layers, batch, self.num_cells, device=device)
        output, (hn, cn) = self.rnn(inputs, (h0, c0))

        # output is (seq_len, batch, num_cells)
        output = output.transpose(0, 1)

        # output is (batch, seq_len, num_cells)
        output = self.classifier(output)
        # output is (batch, seq_len, vocab_size)
        return output

    def sample(self, x, length):
        with torch.no_grad():
            # x is (seq_len, )
            if len(x.shape) != 1:
                print("""We expect just a one dimensional array for the input when
                      sampling""")
            batch = 1
            # inputs is (seq_len, batch, vocab_size)
            device = x.device
            generated_seq = []
            h0 = c0 = torch.zeros(self.num_layers, batch, self.num_cells, device=device)
            next_char_idx = None
            # Feed the network with the starting tensor
            charidxs = np.arange(self.vocab_size)
            for xtidx in x:
                xt = torch.zeros(1, 1, self.vocab_size, device=device)
                xt[0, 0, xtidx] = 1
                xt = self.embedding(xt.view(1, -1)).reshape(1, 1,self.num_embeddings)
                output, (h0, c0) = self.rnn(xt, (h0, c0))
                output = output.transpose(0, 1)
                output = self.classifier(output).view(-1)
                # probs may not actually perfectly sum to 1
                probs = F.softmax(output, dim=0).cpu().numpy()
                probs = probs/probs.sum()
                # next_char_idx is used only when leaving the loop
                next_char_idx = np.random.choice(charidxs, p=probs)

            generated_seq.append(next_char_idx)
            for it in range(length):
                xt = torch.zeros(1, 1, self.vocab_size, device=device)
                xt[0, 0, next_char_idx] = 1
                xt = self.embedding(xt.view(1, -1)).reshape(1, 1,self.num_embeddings)
                output, (h0, c0) = self.rnn(xt, (h0, c0))
                output = output.transpose(0, 1)
                output = self.classifier(output).view(-1)
                # probs may not actually perfectly sum to 1
                probs = F.softmax(output, dim=0).cpu().numpy()
                probs = probs/probs.sum()
                # next_char_idx is used only when leaving the loop
                next_char_idx = np.random.choice(charidxs, p=probs)
                generated_seq.append(next_char_idx)
            generated_seq = torch.LongTensor(generated_seq).to(device)
            return torch.hstack((x, generated_seq))
