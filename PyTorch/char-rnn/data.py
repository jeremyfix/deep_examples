#!/usr/bin/env python3
# coding: utf-8

import logging
import torch


def chunks(collection, chunk_size):
    for i in range(0, len(collection), chunk_size):
        yield collection[i:i+chunk_size]


class Dataset():

    def __init__(self, slength):
        self.logger = logging.getLogger()
        # Load the data and build up the conversion map
        self.logger.info("Loading the data")
        text = open('fables.txt').read()
        chars = list(set(text))
        self.idx2char = [chr(ic) for ic in sorted([ord(c) for c in chars])]
        # Add a start of line character
        self.idx2char.append('¤')
        self.char2idx = {char: idx for idx, char in enumerate(self.idx2char)}

        self.logger.info(f"The conversion map is {self.char2idx}")
        # Split the text in non overlapping slength long sentences
        # Note: we use list comprehensions instead of lazy maps
        # because we need explicit constructions

        def str2idx(mystr): return [self.char2idx[c] for c in mystr]
        # Note: we drop the last piece since it may be shorter
        chunked_text = [str2idx(mystr) for mystr in chunks(text, slength)][:-1]

        self.y = torch.Tensor(chunked_text).long()
        # X is the same y, shifted to right and prependend with the
        # start of line character
        self.X = torch.roll(self.y, 1, 1)
        self.X[:, 0] = -1

    @property
    def vocab_size(self):
        return len(self.idx2char)

    def decode(self, stridx):
        return "".join([self.idx2char[ci] for ci in stridx])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]


if __name__ == '__main__':
    import random
    logging.basicConfig()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ds = Dataset(30)
    ith = random.randint(0, len(ds))
    X, y = ds[ith]
    logger.info(f"The {ith}-th input is\n{X}\n, corresponding output is \n{y}")
    logger.info(f"In plain text, y corresponds to \n>>>\n{ds.decode(y)}\n<<<")
