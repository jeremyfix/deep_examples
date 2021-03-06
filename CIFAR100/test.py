from keras.models import load_model
from keras.datasets import cifar100
from keras.utils import to_categorical
import sys

if len(sys.argv) != 2:
    print("Usage : {} model.h5".format(sys.argv[0]))
    sys.exit(-1)

print("Loading the dataset")
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

num_classes = 100
y_test = to_categorical(y_test, num_classes=num_classes)

print("Loading the model")
model = load_model(sys.argv[1])

print("Evaluating the model")

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
