
##############################################################
#### Simulations inspired from "All you need is a good init"

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.datasets import cifar100
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Lambda, Dense, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD

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

for i in range(3):
    x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer=kernel_initializer)(x)

for i in range(2):
    x = Conv2D(filters=48, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer=kernel_initializer)(x)

x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

for i in range(5):
    x = Conv2D(filters=80, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer=kernel_initializer)(x)

x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

for i in range(5):
    x = Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer=kernel_initializer)(x)

x = GlobalAveragePooling2D()(x)
x = Dense(500, activation='relu', kernel_initializer=kernel_initializer)(x)
yo = Dense(num_classes, activation='softmax', kernel_initializer=kernel_initializer)(x)

model = Model(inputs=[xi], outputs=[yo])
optimizer = SGD(lr=0.01, momentum=0.9)

def lr_rate(epoch):
    base_lr = 1e-3
    if(epoch <= 100):
        return base_lr
    elif(epoch <= 150):
        return base_lr/1e1
    elif(epoch <= 200):
        return base_lr/1e2
    else:
        return base_lr/1e3
    
lr_sched = LearningRateScheduler(lr_rate)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train,\
                    epochs=230,\
                    batch_size=32, \
                    validation_data=(x_val, y_val),
                    callbacks=[lr_sched])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.figure()

plt.subplot(121)
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig('fitnet.pdf', bbox_inches='tight')
