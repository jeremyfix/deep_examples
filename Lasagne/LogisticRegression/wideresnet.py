import matplotlib.pyplot as plt

import sys
import os

import mnist

import lasagne
import theano
import theano.tensor as T

import numpy as np 


minibatch_size = 600
nb_training_samples = 60000
nb_test_samples = 10000
max_epoch = 80
momentum = 0.9
weight_decay = 0.0001

def iterate_minibatches(inputs, targets, batchsize):
    p = np.random.permutation(inputs.shape[0])
    for idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        yield (inputs[p])[idx:idx+batchsize],(targets[p])[idx:idx+batchsize]

'''
    blocks_size : a list with the number of filters for the 2 convolutions of a block
'''
def build_model(blocks_size):
    input_var = T.tensor4('inputs')
    n_params = 0
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    l_prev = l_in
    for k in blocks_size:
        l_prev = lasagne.layers.Conv2DLayer(l_prev, k, 3, nonlinearity=lasagne.nonlinearities.rectify)
        n_params += lasagne.layers.count_params(l_prev)
        l_prev = lasagne.layers.Conv2DLayer(l_prev, k, 3, nonlinearity=lasagne.nonlinearities.linear)
        n_params += lasagne.layers.count_params(l_prev)
        #
        # TODO : add the x in H(x) = x + F(x)
        #
    l_out = lasagne.layers.DenseLayer(l_prev, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
    n_params += lasagne.layers.count_params(l_out)
    print("%i parameters to find."% n_params)
    return input_var, l_out


print("Loading the data")
X_train, y_train, X_test, y_test = mnist.load_dataset(nb_training_samples, nb_test_samples, False)

print("Building the model")
input_var, model = build_model((3,3))
prediction = lasagne.layers.get_output(model)

# Define the Negative Log Likelihood loss
target_var = T.ivector('targets')
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

l2 = lasagne.regularization.regularize_network_params(model, lasagne.regularization.l2)
loss += weight_decay * l2

# Define the update of the weights
params = lasagne.layers.get_all_params(model, trainable=True)
updates = lasagne.updates.adagrad(loss, params)
updates = lasagne.updates.apply_momentum(updates, momentum=momentum)

# Define the function to compute the test loss and test accuracy
accuracy = lasagne.objectives.categorical_accuracy(prediction, target_var)
accuracy = accuracy.mean()

print("Compiling the theano function")
train_fn = theano.function([input_var, target_var], loss, updates=updates)
accu_fn = theano.function([input_var, target_var], accuracy)
loss_fn = theano.function([input_var, target_var], loss)

epoch = 0

train_loss_arr = []
train_accu_arr = []
test_loss_arr = []
test_accu_arr = []

train_accu_arr.append(accu_fn(X_test, y_test))
test_accu_arr.append(accu_fn(X_test, y_test))
train_loss_arr.append(loss_fn(X_test, y_test))
test_loss_arr.append(loss_fn(X_test, y_test))

for i in range(max_epoch):
    for Xi, yi in iterate_minibatches(X_train, y_train, minibatch_size):
        sys.stdout.write('\rEpoch : %f ' % epoch)
        train_fn(Xi, yi)
        epoch += 1. / (nb_training_samples/float(minibatch_size))
    train_accu_arr.append(accu_fn(X_train, y_train))
    test_accu_arr.append(accu_fn(X_test, y_test))
    train_loss_arr.append(loss_fn(X_train, y_train))
    test_loss_arr.append(loss_fn(X_test, y_test))

print("")

print(" Test accuracy : %f\n" % test_accu_arr[-1])

plt.figure()
plt.subplot(1,2,1)
plt.plot(np.arange(max_epoch+1), train_accu_arr, 'b-', label='Train accuracy')
plt.plot(np.arange(max_epoch+1), test_accu_arr, 'r-', label='Test accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(np.arange(max_epoch+1), train_loss_arr, 'b-', label='Train loss')
plt.plot(np.arange(max_epoch+1), test_loss_arr, 'r-', label='Test loss')
plt.legend()

plt.show()

