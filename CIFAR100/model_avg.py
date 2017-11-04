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
    loss = -np.log(logits[np.arange(y_classes.shape[0]),y_classes.ravel()]).sum()/y_classes.shape[0]
    pred_classes = np.argmax(logits, axis=1)
    acc = (pred_classes == y_classes.ravel()).sum()/y_classes.shape[0]
    return loss, acc
                   

print("Loading the models and computing their predictions")
predictions = []
for idx, f in enumerate(sys.argv[1:]):
    model = load_model(f)

    print("Computing the predictions of {}".format(f))

    pred = model.predict(x_test,verbose=0)

    scores = acc_loss(pred, y_test)
    print("Model {} : Loss={}, Accuracy={}".format(idx, scores[0], scores[1]))
    predictions.append(pred)
    del model
    
predictions = np.array(predictions)
pred_avg = np.mean(predictions, axis=0)
print(pred_avg.shape)
scores = acc_loss(pred, y_test)
print("Avg Model : Loss={}, Accuracy={}".format(scores[0], scores[1]))
