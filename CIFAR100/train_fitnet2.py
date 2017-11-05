
##############################################################
#### Simulations inspired from "All you need is a good init"
## From the original paper, they said to use lr=0.01 at start
## but after 30 steps, the loss diverges.


import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import argparse

from keras.datasets import cifar100
from keras.layers import Input, Lambda, Dense, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler, Callback
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add
from keras import regularizers

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_augment',
    help='Specify if you want to use data augmentation',
    action='store_true'
)

parser.add_argument(
    '--dropout',
    help='Specify to use dropout',
    action='store_true'
)
parser.add_argument(
    '--shortcut',
    help='Specify to use shortcut connections',
    action='store_true'
)

parser.add_argument(
    '--l2_reg',
    help='Specify to use l2_reg',
    action='store_true'
)

parser.add_argument(
    '--bn',
    help='Specify to use batch normalization',
    action='store_true'
)

parser.add_argument(
    '--base_lrate',
    help='Which base learning rate to use',
    type=float,
    default=0.001
)

parser.add_argument(
    '--activation',
    help="which activation function to use",
    choices=['relu','elu'],
    required=True,
    action='store'
)

parser.add_argument(
    '--batch_size',
    required=True,
    type=int
)

parser.add_argument(
    '--runid',
    required=True, 
    type=int
)

args = parser.parse_args()

use_dataset_augmentation = args.data_augment
use_dropout = args.dropout
use_shortcut = args.shortcut
use_bn = args.bn
base_lrate = args.base_lrate
activation = args.activation
run_id = args.runid
batch_size = args.batch_size
use_l2_reg = args.l2_reg

print("Loading the dataset")
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

def split(X, y, test_size):
    '''
    X : 4D tensor images
    y : 2D tensor labels
    test_size : in [0, 1] 
    '''
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    nb_test = int(test_size * X.shape[0])
    return X[nb_test:,:, :, :], y[nb_test:, :],\
        X[:nb_test, :, :, :], y[:nb_test, :]

print("Splitting the dataset")
x_train, y_train, x_val, y_val = split(x_train, y_train,
                                       test_size=0.2)

print("I loaded {} training images of size {} x {} x {}".format(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3]))
print("I loaded {} validation images of size {} x {} x {}".format(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3]))
print("I loaded {} test images of size {} x {} x {}".format(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3]))

x_train_mean = x_train.mean(axis=0)
x_train_std = x_train.std(axis=0)

input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
img_height = x_train.shape[1]
img_width = x_train.shape[2]
num_classes = 100

y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

################# Building up the model
xi = Input(shape=input_shape, name="input")
xl = Lambda(lambda image, mu, std: (image-mu)/std,
            arguments={'mu': x_train_mean,
                       'std': x_train_std})(xi)

x = xl

### FitNet-4
# kernel_initializer='glorot_uniform' does not work very well
# kernel_initializer='glorot_normal'
kernel_initializer='he_normal'
if use_l2_reg:
    kernel_regularizer=regularizers.l2(0.0005)
else:
    kernel_regularizer=None

x_prev = x
for i in range(2):
    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
if use_shortcut:
    x_prev = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x_prev)
    x_prev = Activation(activation)(x_prev)
    x = add([x_prev, x])

x_prev = x
for i in range(2):
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
if use_shortcut:
    x_prev = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x_prev)
    x_prev = Activation(activation)(x_prev)
    x = add([x_prev, x])

x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

x_prev = x
for i in range(2):
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
if use_shortcut:
    x_prev = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x_prev)
    x_prev = Activation(activation)(x_prev)
    x = add([x_prev, x])

x = GlobalAveragePooling2D()(x)
x = Dense(500, activation=activation, kernel_initializer=kernel_initializer)(x)
if use_dropout:
    x = Dropout(0.5)(x)
yo = Dense(num_classes, activation='softmax', kernel_initializer=kernel_initializer)(x)

model = Model(inputs=[xi], outputs=[yo])
optimizer = SGD(lr=0.01, momentum=0.9)

def lr_rate(epoch):
    if(epoch <= 50):
        return base_lrate
    elif(epoch <= 100):
        return base_lrate/1e1
    elif(epoch <= 150):
        return base_lrate/1e2
    
lr_sched = LearningRateScheduler(lr_rate)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_train_begin(self, logs={}):
        self.test_history = {'loss':[], 'acc':[]}
        
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.test_history['loss'].append(loss)
        self.test_history['acc'].append(acc)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

test_cb = TestCallback((x_test, y_test))

if use_dataset_augmentation:
    # With data augmentation
    datagen = ImageDataGenerator(width_shift_range=5./img_width,
                                 height_shift_range=5./img_height,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=len(x_train)/batch_size,
                                  epochs =230,
                                  validation_data=(x_val, y_val),
                                  callbacks=[lr_sched, test_cb])
else:
    # Without data augmentation
    history = model.fit(x_train, y_train,\
                        epochs=150,\
                        batch_size=batch_size, \
                        validation_data=(x_val, y_val),
                        callbacks=[lr_sched, test_cb])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.figure()

plt.subplot(121)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(test_cb.test_history['acc'])
for e in [50, 100]:
    plt.axvline(e, linewidth=2, linestyle='--', color='r')

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val','test'], loc='center right')

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(test_cb.test_history['loss'])
for e in [100, 150, 200]:
    plt.axvline(e, linewidth=2, linestyle='--', color='r')
    
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val', 'test'], loc='center right')

suptitle = "v2 "
if use_dataset_augmentation:
    suptitle += " DatasetAugment "
if use_dropout:
    suptitle += " Dropout"
if not use_bn:
    suptitle += " NoBN "
suptitle += " lr:{} ".format(base_lrate)
suptitle += " bs:{} ".format(batch_size)
suptitle += activation
suptitle += " shortcut "
suptitle += '_' + str(run_id)
plt.suptitle(suptitle)
filename = 'fitnet' + suptitle.replace(' ','_')
plt.savefig(filename + ".pdf", bbox_inches='tight')


model.save('weights/'+ filename + '.h5')
