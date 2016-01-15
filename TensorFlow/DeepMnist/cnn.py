# Convolutional neural networks for the MNIST dataset

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sys

# Load the dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

mnist_img_size = 784
mnist_nb_labels = 10
mini_batch_size = 100

##############################
# Define the computation graph

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


## Some handy functions to create the convolution and pooling layers
# Stride has dim 4 : num_samples x height x width x num_channels
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

