# TODO :
# save the learning curves with a small description at the beginning of the setting


import matplotlib.pyplot as plt

import sys
import os

import mnist

import lasagne
import theano
import theano.tensor as T
import LogisticRegression
import numpy as np 


import argparse


################################################################
# Argument parsing
################################################################

description = '''
This program allows to test different neural network architectures as well as learning algorithms on the MNIST classification problem.\n
The parameters of the models / learning algorithms must be provided as a comma separated list. Depending on the argument, a different number of parameters must be provided\n
For example, you can use it calling :\n

    For a logistic regression and a adagrad learning algorithm and a common learning rate of 1e-1\n
    %s --logreg --adagrad 1e-1
''' % sys.argv[0]
parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)

modeltype = parser.add_mutually_exclusive_group(required=True)
modeltype.add_argument('--logreg', action='store_true', help='Logistic regression model')
modeltype.add_argument('--cnn', type=str, help='Convutional Neural Network model')
modeltype.add_argument('--vgg', type=str, help='Oxford vision group model')

learning_algo = parser.add_mutually_exclusive_group(required=True)
learning_algo.add_argument('--sgd', type=str, help='Stochastic gradient descent')
learning_algo.add_argument('--adagrad', type=str, help='Adaptive gradient')
learning_algo.add_argument('--rmsprop', type=str, help='RMSprop')

parser.add_argument('--momentum', type=float, help='Momentum value')
parser.add_argument('--nesterov_momentum', type=float, help='Nesterov momentum value')
parser.add_argument('--l2', type=float, help='L2 regularization')
parser.add_argument('--l1', type=float, help='L1 regularization')

parser.add_argument('--nbtrain', type=int, help='Number of training samples', default=60000)
parser.add_argument('--nbtest', type=int, help='Number of test samples', default=10000)
parser.add_argument('--batchsize', type=int, help='Minibatch size', default=600)
parser.add_argument('--epoch', type=int, help='Number of training epochs', default=80)

# Parse the arguments and perform some clean up dropping out the None values
args = {k:v for k, v in vars(parser.parse_args()).iteritems() if v}
print(args)

################################################################


def iterate_minibatches(inputs, targets, batchsize):
    p = np.random.permutation(inputs.shape[0])
    for idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        yield (inputs[p])[idx:idx+batchsize],(targets[p])[idx:idx+batchsize]

print("Loading the data")
X_train, y_train, X_test, y_test = mnist.load_dataset(args['nbtrain'], args['nbtest'], False)

############################################################################
# Model building

print("Model architecture : ")
filename_prefix = ""
if('logreg' in args):
    print("    Logistic regression, Input -> Softmax")
    model = LogisticRegression.Model(28, 28, 1, 10)
    filename_prefix += "logreg_"
elif 'mlp' in args:
    print("    Multilayer perceptron with : ")
    filename_prefix += "mlp_"
    sys.exit()
elif 'cnn' in args:
    print("    Convolutional neural network with :")
    filename_prefix += "cnn_"
    sys.exit()
elif 'vgg' in args:
    print("    Oxford Visual Geometry Group with :")
    filename_prefix += "vgg_"
    sys.exit()
else:
    sys.exit()
print("")

input_var, l_out = model.get_input_var(), model.get_output_layer()
prediction = lasagne.layers.get_output(l_out)

# Define the Negative Log Likelihood loss
target_var = T.ivector('targets')
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

print("Regularization :")
if 'l2' in args:
    print("    L2 regularization of lambda = %f" % args['l2'])
    l2 = lasagne.regularization.regularize_network_params(l_out, lasagne.regularization.l2)
    loss += args['l2'] * l2
    filename_prefix += "l2_%f" % args['l2']
elif 'l1' in args:
    print("    L1 regularization of lambda = %f " % args['l1'])
    l1 = lasagne.regularization.regularize_network_params(l_out, lasagne.regularization.l1)
    loss += args['l1'] * l1
    filename_prefix += "l1_%f" % args['l1']
else:
    print("    No regularizatiion")

# Define the update of the weights
params = lasagne.layers.get_all_params(l_out, trainable=True)

print("Learning algorithm :")
if 'sgd' in args:
    learning_rate = float(args['sgd'])
    print("    Stochastic Gradient descent")
    print("      * Learning rate : %f" % learning_rate)
    updates = lasagne.updates.sgd(loss, params, learning_rate)
    filename_prefix += "sgd_lr%f_" % learning_rate
elif 'adagrad' in args:
    learning_rate = float(args['adagrad'])
    print("    Adaptive Gradient descent")
    print("      * Learning rate : %f" % learning_rate)
    momentum = 0.9
    if(momentum in args):
        momentum = args['momentum']
    print("      * Momentum : %f " % momentum)
    updates = lasagne.updates.adagrad(loss, params, learning_rate, momentum)
    filename_prefix += "adagrad_lr%f_m%f_" % (learning_rate,momentum)
elif 'rmsprop' in args:
    learning_rate, rho, epsilon = map(float, args['rmsprop'].split(','))
    print("    RMS prop : ")
    print("      * Learning rate : %f" % learning_rate)
    print("      * rho           : %f" % rho)
    print("      * epsilon       : %f" % epsilon)
    updates = lasagne.updates.rmsprop(loss, params, learning_rate, rho, epsilon)
    filename_prefix += "rmsprop_lr%f_rho%f_eps%f_" % (learning_rate, rho, epsilon)
print("")


# Momentum ?
if 'momentum' in args and not 'adagrad' in args:
    print("Applying a momentum of %f " % args['momentum'])
    updates = lasagne.updates.apply_momentum(updates, momentum=args['momentum'])
    filename_prefix += "m%f_" % args['momentum']
elif 'nesterov_momentum' in args and not 'adagrad' in args:
    print("Applying a Nesterov momentum of %f " % args['nesterov_momentum'])
    updates = lasagne.updates.apply_nesterov_momentum(updates, momentum=args['nesterov_momentum'])
    filename_prefix += "nm%f_" % args['nesterov_momentum']

print("\n Filename prefix : %s " % filename_prefix)
print("\n  ==>  I have %i parametes to update  <== \n" % lasagne.layers.count_params(l_out))

# Define the function to compute the test loss and test accuracy
accuracy = lasagne.objectives.categorical_accuracy(prediction, target_var)
accuracy = accuracy.mean()

print("Compiling the theano function")
train_fn = theano.function([input_var, target_var], loss, updates=updates)
accu_fn = theano.function([input_var, target_var], accuracy)
loss_fn = theano.function([input_var, target_var], loss)

epoch = 0
train_accu, test_accu = 0, 0

file_loss_acc = open("%s.data" % filename_prefix, 'w', 0) # unbuffered
for i in range(args['epoch']):
    for Xi, yi in iterate_minibatches(X_train, y_train, args['batchsize']):
        sys.stdout.write('\rEpoch : %f ' % epoch)
        train_fn(Xi, yi)
        epoch += 1. / (args['nbtrain']/float(args['batchsize']))
    train_accu = accu_fn(X_train, y_train)
    test_accu = accu_fn(X_test, y_test)
    train_loss = loss_fn(X_train, y_train)
    test_loss = loss_fn(X_test, y_test)
    file_loss_acc.write("%i %f %f %f %f\n" % (epoch, train_loss, train_accu, test_loss, test_accu))
file_loss_acc.close()


print(" Train accuracy : %f\n" % train_accu)
print(" Test accuracy : %f\n" % test_accu)

