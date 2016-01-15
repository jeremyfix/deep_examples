# *-* coding: utf-8 *-*
# Convolutional neural networks for the MNIST dataset

# QUestions : 
#    why the dims weight_variable([5, 5, 1, 32])
#       the input tensor is batch_size x height x width x nb_channels
#    why the dims x_image = tf.reshape(x, [-1,28,28,1]
# 

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sys

# Load the dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

mnist_width = 28
mnist_height = 28
mnist_img_size = mnist_width * mnist_height
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

#### Definition of the architecture
# width x height x nb channels in x nb channels out

x = tf.placeholder(tf.float32, [None, mnist_img_size])
y_ = tf.placeholder(tf.float32, [None, mnist_nb_labels])

L1_patch_size = 5
L1_chan_out = 32

L2_patch_size = 5
L2_chan_out = 64


# Layer 1
W_conv1 = weight_variable([L1_patch_size, L1_patch_size, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,mnist_width,mnist_height,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Layer 2
W_conv2 = weight_variable([L2_patch_size, L2_patch_size, L1_chan_out, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess = tf.Session()
sess.run(tf.initialize_all_variables())

Nsteps = 20000
for i in range(Nsteps):
    sys.stdout.write("\r Step %i / %i " % (i+1, Nsteps))
    sys.stdout.flush()
    batch = mnist.train.next_batch(mini_batch_size)
    if i%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        sys.stdout.write("\r step %d, training accuracy %g\n"%(i, train_accuracy))
        sys.stdout.flush()
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"% sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
