from keras.datasets import cifar100
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Lambda, Dense, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.utils import to_categorical

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
for i in range(2):
    x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

for i in range(2):
    x = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

for i in range(2):
    x = Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
xf = GlobalAveragePooling2D()(x)
xo = Dense(num_classes, name="y")(xf)
yo = Activation('softmax', name="y_act")(xo)
model = Model(inputs=[xi], outputs=[yo])
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,\
          epochs=32,\
          batch_size=32, \
          validation_data=(x_val, y_val))
