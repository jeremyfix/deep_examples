
import sys
import os

import mnist

import lasagne
import theano
import theano.tensor as T

import numpy as np 

minibatch_size = 50
nb_training_samples = 2000
nb_test_samples = 1000
max_epoch = 20

def iterate_minibatches(inputs, targets, batchsize):
    p = np.random.permutation(inputs.shape[0])
    for idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        yield (inputs[p])[idx:idx+batchsize],(targets[p])[idx:idx+batchsize]

def build_model():
    input_var = T.tensor4('inputs')
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    l_out = lasagne.layers.DenseLayer(l_in, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
    return input_var, l_out


print("Loading the data")
X_train, y_train, X_test, y_test = mnist.load_dataset(nb_training_samples, nb_test_samples)

print("Building the model")
input_var, model = build_model()
prediction = lasagne.layers.get_output(model)

# Define the Negative Log Likelihood loss
target_var = T.ivector('targets')
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

# Define the update of the weights
params = lasagne.layers.get_all_params(model, trainable=True)
updates = lasagne.updates.adagrad(loss, params)

# Define the function to compute the test loss and test accuracy


print("Compiling the theano function")
train_fn = theano.function([input_var, target_var], loss, updates=updates)

epoch = 0
for i in range(max_epoch):
    for Xi, yi in iterate_minibatches(X_train, y_train, minibatch_size):
        sys.stdout.write('\rEpoch : %f ' % epoch)
        train_fn(Xi, yi)
        epoch += 1. / (nb_training_samples/float(minibatch_size))

print("")
