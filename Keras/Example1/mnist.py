# In this script we train a simple CNN network
# and visualize the learned features

## TODO :
# - subsample the training set for finding the architectures which allows overfitting (np.random.choice)
# - preprocess the input ? no, use batchnorm
# - visualize the first filters
# - if batchnorm is used, checkout its parameters
# - visualize the activation of the units function of the image index
# - export the model size
# - perform a t-SNE on the activations
# - deconvolution of the hidden layer units ? what is the input that will maximally activate a hidden unit ?


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.utils import np_utils

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

###### Definition of the network


def build_network(cnn_layer_sizes, fc_layer_sizes):
    '''The network is built from
    a stack of convolutive - max pooling layers
    followed by fully connected layers
    The output is a softmax, the loss is the cross-entropy
    '''
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Convolution2D(cnn_layer_sizes[0], 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))
    for s in cnn_layer_sizes[1:]:
		#model.add(BatchNormalization())
        model.add(Convolution2D(s, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))
    model.add(Flatten())
    for k, s in enumerate(fc_layer_sizes):
		#model.add(BatchNormalization())
        model.add(Dense(s))
        if(k == (len(fc_layer_sizes)-1)):
            model.add(Activation('softmax'))
        else:
            model.add(Activation('relu'))
    return model


#### Building the network

model = build_network([32, 16],[50, num_classes])
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#### Fitting the network
batch_size = 32
nb_epoch = 100
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                 verbose=1, validation_data=(X_test, Y_test))
print(hist.history)

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

#### Evaluating the network
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
