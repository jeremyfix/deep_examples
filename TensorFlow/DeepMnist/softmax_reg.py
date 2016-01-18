# Softmax regression for the MNIST dataset
# https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

# It monitors the empirical and validation accuracies during training
# and plot them at the end
# You also get the test accuracy at the end, something like
# Empirical : 0.917236, Validation : 0.923000 , Test : 0.914500

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sys
import numpy as np

# Load the dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

mnist_width = 28
mnist_height = 28
mnist_img_size = mnist_width * mnist_height
mnist_nb_labels = 10
mini_batch_size = 100

##############################
# Define the computation graph
x = tf.placeholder(tf.float32, [None, mnist_img_size])
W = tf.Variable(tf.zeros((mnist_img_size, mnist_nb_labels)))
b = tf.Variable(tf.zeros((mnist_nb_labels,)))

# Predicted probabilities
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Desired output
y_ = tf.placeholder(tf.float32, [None, mnist_nb_labels])

# The cross entropy loss
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

################################
# Initialization operation

init = tf.initialize_all_variables()

################################
# Evaluation operations
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


###############################
# Model training

sess = tf.Session()
sess.run(init)

empirical_accuracies = []
validation_accuracies = []
Nsteps = 1000
for i in range(Nsteps):
    sys.stdout.write("\r Step %i / %i " % (i+1, Nsteps))
    sys.stdout.flush()
    batch_xs, batch_ys = mnist.train.next_batch(mini_batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Evalute the accuracy on the validation set
    empirical_accuracies.append(sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
    validation_accuracies.append(sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))
sys.stdout.write('\n')

import matplotlib.pyplot as plt
plt.figure()
plt.plot(validation_accuracies, 'b', label='validation')
plt.plot(empirical_accuracies, 'r', label='empirical')
plt.legend()
#plt.show()

##################################
# Model evaluation
empirical_accuracy = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
validation_accuracy = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Accuracies : \n Empirical : %f, Validation : %f , Test : %f\n" %(empirical_accuracy, validation_accuracy, test_accuracy))

####################################
###### visualisation of the weights
weights = sess.run(W)
biases = sess.run(b)

# We now plot the 10 weights
import matplotlib.gridspec as gridspec
nc = int(np.sqrt(weights.shape[1]))
nr = int(np.ceil(float(weights.shape[1]) / nc))
gs = gridspec.GridSpec(nr, nc)


vmin = -np.max(map(np.abs, [weights.min(), weights.max()]))
vmax = np.max(map(np.abs, [weights.min(), weights.max()]))
print("Weights will be shown normalized in [%f, %f]" % (vmin, vmax))

plt.figure()
w_index = 0
for i in range(nr):
    for j in range(nc):
        if(w_index >= weights.shape[1]):
            break
        ax = plt.subplot(gs[i,j])
        ax.imshow(weights[:,w_index].reshape((mnist_height, mnist_width)),vmin=vmin, vmax=vmax, cmap='gray')
        w_index += 1

plt.show()
