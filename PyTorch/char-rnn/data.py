#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import logging
import json
# External imports
import torch


def chunks(collection, chunk_size):
    for i in range(0, len(collection), chunk_size):
        if i+chunk_size+1 < len(collection):
            yield collection[i:i+chunk_size], collection[i+1:i+chunk_size+1]

class CharMap():

    def __init__(self, chars=None):
        if chars is not None:
            self.idx2char = [chr(ic) for ic in sorted([ord(c) for c in chars])]
            self._build_char_map()

    def _build_char_map(self):
        self.char2idx = {char: idx for idx, char in enumerate(self.idx2char)}

    @property
    def vocab_size(self):
        return len(self.idx2char)

    def decode(self, stridx):
        return "".join([self.idx2char[ci] for ci in stridx])

    def encode(self, mystr):
        return [self.char2idx[c] for c in mystr]

    def __repr__(self):
        return f"{self.char2idx}"

    @classmethod
    def load(cls, filename):
        chars = list(open(filename, 'r').read())
        charmap = CharMap()
        charmap.idx2char = chars
        charmap._build_char_map()
        return charmap

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write("".join(self.idx2char))

class Dataset():

    def __init__(self, slength):
        self.logger = logging.getLogger()
        # Load the data and build up the conversion map
        self.logger.info("Loading the data")
        text = open('fables.txt').read()
        chars = list(set(text))
        self._charmap = CharMap(chars)
        self._charmap.save('charmap')

        self.logger.info(f"The conversion map is {self._charmap}")
        # Split the text in non overlapping slength long sentences
        # Note: we use list comprehensions instead of lazy maps
        # because we need explicit constructions

        # Note: we drop the last piece since it may be shorter
        x_sentences = []
        y_sentences = []
        for chunk_x, chunk_y in chunks(text, slength):
            x_sentences.append(self.charmap.encode(chunk_x))
            y_sentences.append(self.charmap.encode(chunk_y))

        self.X = torch.LongTensor(x_sentences)
        self.y = torch.LongTensor(y_sentences)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]

    @property
    def charmap(self):
        return self._charmap


if __name__ == '__main__':
    import random
    logging.basicConfig()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ds = Dataset(30)
    charmap = ds.charmap
    ith = random.randint(0, len(ds))
    X, y = ds[ith]
    logger.info(f"The {ith}-th input is\n{X}\ncorresponding output is \n{y}")
    logger.info(f"In plain text, y corresponds to \n>>>\n{charmap.decode(y)}\n<<<")
