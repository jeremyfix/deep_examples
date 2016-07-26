
import mnist

import lasagne
import theano
import theano.tensor as T

X_train, y_train, X_test, y_test = mnist.load_dataset(2000, 1000)


