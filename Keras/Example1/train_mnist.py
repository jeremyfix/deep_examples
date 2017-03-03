# In this script we train a simple CNN network
# and visualize the learned features

from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
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


def build_network(input_shape):
    '''The network is built from
    a stack of convolutive - max pooling layers
    followed by fully connected layers
    The output is a softmax, the loss is the cross-entropy
    '''
    init_conv = 'glorot_uniform'
    filter_shape = (3, 3)

    model = Sequential()
        
    model.add(Convolution2D(32, filter_shape[0], filter_shape[1], init=init_conv, border_mode='same', input_shape=input_shape, name='conv1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, filter_shape[0], filter_shape[1], init=init_conv, border_mode='same', name='conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', name='max1'))
    
    model.add(Convolution2D(16, filter_shape[0], filter_shape[1], init=init_conv, border_mode='same', name='conv3'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, filter_shape[0], filter_shape[1], init=init_conv, border_mode='same', name='conv4'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', name='max2'))
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(128, name="d1"))
    model.add(Activation('relu'))
    model.add(Dense(64, name="d2"))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

def build_res_network(input_shape):
    '''The network is built from
    a stack of convolutive - max pooling layers
    some split and fusion are operated over each conv conv pool blocks
    followed by fully connected layers
    The output is a softmax, the loss is the cross-entropy
    '''
    init_conv = 'glorot_uniform'
    filter_shape = (3, 3)

    input = Input(shape=input_shape)

    K = 32

    l1 = input
    l2 = Convolution2D(K, filter_shape[0], filter_shape[1], init=init_conv, border_mode='same', input_shape=input_shape, name='conv1')(l1)
    l2  = Activation('relu')(l2)
    l2 = Convolution2D(K, filter_shape[0], filter_shape[1], init=init_conv, border_mode='same', name='conv2')(l2)
    l2 = Activation('relu')(l2)
    l2 = MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', name='max1')(l2)

    # Handle the shortcut.
    # We need to add layers for expanding the number of channels of
    # the input of the block
    l1 = MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid')(l1)
    l1 = Convolution2D(K, 1, 1, border_mode='valid')(l1)
    l1 = merge([l1, l2], mode="sum")

    l1 = GlobalAveragePooling2D()(l1)
    
    l1 = Dense(128, name="d1")(l1)
    l1 = Activation('relu')(l1)
    l1 = Dense(64, name="d2")(l1)
    l1 = Activation('relu')(l1)
    l1 = Dense(10)(l1)
    l1 = Activation('softmax')(l1)

    output = l1
    
    model = Model(input=input, output=output)
    return model

#### Building the network

model = build_res_network(input_shape)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#### Fitting the network
batch_size = 32
nb_epoch = 5
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                 verbose=1, validation_data=(X_test, Y_test))
print(hist.history)

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

#### Evaluating the network
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
