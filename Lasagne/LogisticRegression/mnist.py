import sys
import os

import numpy as np
import theano
import theano.tensor as T

import lasagne

''' 
    Download and loads the mnist dataset
    Adapted from Lasagne/mnist.py to support normalization
    Returns X_train, y_train, X_test, y_test
    with X_train a (nb_samples, nb_channels, width, height) tensor, here nb_channels =1
     and y_train a (nb_samples,) tensor
'''
def load_dataset(nb_train=60000, nb_test=10000, normalize=True):
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        else:
            print("Dataset %s already downloaded" % filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')[:min(nb_train, 60000)]
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')[:min(nb_train, 60000)]
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')[:min(nb_test, 10000)]
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')[:min(nb_test, 10000)]

    if(normalize):
        X_mean = X_train[:,0,:,:].mean(axis=0)
        X_std = X_train[:,0,:,:].std(axis=0)
        X_train -= X_mean
        # In order to use elementwise division with broadcasting,
        # we first replace all the null std by 1.
        X_std[X_std == 0] = 1.
        X_train /= X_std

        X_test -= X_mean
        X_test /= X_std

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_test, y_test
