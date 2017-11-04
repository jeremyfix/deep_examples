from keras.models import load_model
from keras.datasets import cifar100
from keras.utils import to_categorical
import numpy as np
import sys

if len(sys.argv) <= 1:
    print("Usage : {} model1.h5 model2.h5 ...".format(sys.argv[0]))
    sys.exit(-1)

print("Loading the dataset")
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

num_classes = 100
y_test_cat = to_categorical(y_test, num_classes=num_classes)

def acc_loss(logits, y_classes):
    loss = -np.log(logits[:,y_classes]).sum()
    return loss
                   

print("Loading the models and computing their predictions")
predictions = []
for f in sys.argv[1:]:
    model = load_model(f)

    print("Computing the predictions of {}".format(f))

    pred = model.predict(x_test,verbose=0)
    predictions.append(pred)
    scores = model.evaluate(x_test, y_test_cat, verbose=0)
    print("From eval : {}".format(scores))
    scores = acc_loss(pred, y_test)
    print(scores)
    
predictions = np.array(predictions)
print(predictions.shape)
print(predictions.dtype)
