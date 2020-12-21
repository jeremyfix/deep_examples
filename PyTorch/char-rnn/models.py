
# Standard imports
# External imports
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, vocab_size, num_cells, num_layers, num_hidden):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.num_cells = num_cells
        self.num_layers = num_layers
        self.rnn = nn.LSTM(vocab_size, num_cells, num_layers)
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Linear(num_cells, num_hidden),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(num_cells, vocab_size)
        )


    def forward(self, x):
        # x is (batch, seq_len)
        (batch, seq_len), input_size = x.shape, self.vocab_size

        x_idx = x.view(-1)

        # the inputs to the LSTM must be (seq_len, batch, input_size)
        # we first need to one-hot encode the inputs
        inputs = torch.zeros(seq_len*batch, input_size)
        inputs[x_idx] = 1
        inputs = torch.transpose(inputs.view(batch, seq_len, input_size), 0, 1)

        # input to the LSTM must be (seq_len, batch, input_size)
        # h0, c0 are (num_layers, batch, hidden_size)
        h0 = c0 = torch.zeros(self.num_layers, batch, self.num_cells)
        output, (hn, cn) = self.rnn(inputs, (h0, c0))

        # The output is (batch_size, vocab_size)
        output = self.classifier(output)
        return output
