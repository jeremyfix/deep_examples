from keras.models import load_model
from keras.datasets import cifar100
from keras.utils import to_categorical
import sys

if len(sys.argv) <= 1:
    print("Usage : {} model1.h5 model2.h5 ...".format(sys.argv[0]))
    sys.exit(-1)

print("Loading the dataset")
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

num_classes = 100
y_test = to_categorical(y_test, num_classes=num_classes)

print("Loading the models and computing their predictions")
for f in sys.argv[1:]:
    model = load_model(f)

    print("Computing the predictions of {}".format(f))

    pred = model.predict(x_test,verbose=0)
    print(pred)

