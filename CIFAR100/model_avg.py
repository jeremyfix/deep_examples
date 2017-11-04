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
    print(logits, y_classes)
    print(logits[:, y_classes])
    loss = -np.log(logits[:,y_classes]).sum()/y_classes.shape[0]
    pred_classes = np.argmax(logits, axis=1)
    acc = (pred_classes == y_classes).sum()/y_classes.shape[0]
    return loss, acc
                   

print("Loading the models and computing their predictions")
predictions = []
for f in sys.argv[1:]:
    model = load_model(f)

    print("Computing the predictions of {}".format(f))

    pred = model.predict(x_test,verbose=0)
    scores = model.evaluate(x_test[:2,:,:,:], y_test_cat[:2], verbose=0)
    print("From eval : {}".format(scores))
    scores = acc_loss(pred[:2], y_test[:2])
    print(scores)
    predictions.append(pred)
    
predictions = np.array(predictions)
print(predictions.shape)
print(predictions.dtype)
