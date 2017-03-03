import sys
import keras
from keras.datasets import mnist
from keras import backend as K
from keras.utils import np_utils
from keras.utils.visualize_util import plot


if(len(sys.argv) != 2):
    print("Usage : {} model.hdf5".format(sys.argv[0]))
    sys.exit(-1)

model_fname = sys.argv[1]

###### Some info about the Keras config
if K.image_dim_ordering() == 'th':
    print("Using the theano dim ordering")
    # Batch size x num channels x rows x cols
else:
    print("Using the tensorflow dim ordering")
    # Batch size x rows x cols x num channels

###### Loading the MNIST dataset
# For MNIST, input_shape is (28, 28). The images are monochrome
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_classes = 10
img_rows = 28
img_cols = 28

if K.image_dim_ordering() == 'th':
    # Batch size x num channels x rows x cols
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    # Batch size x rows x cols x num channels
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

##### Loading the model

model = keras.models.load_model(model_fname)
model.summary()
plot(model, to_file= model_fname + '.png', show_shapes=True,show_layer_names=True )

##### Testing the model

score = model.evaluate(X_train, Y_train, verbose=0)
print('Train score:', score[0])
print('Train accuracy: {:%}'.format(score[1]))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy: {:%}'.format(score[1]))
