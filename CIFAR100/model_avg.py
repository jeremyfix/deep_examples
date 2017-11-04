from keras.models import load_model
from keras.datasets import cifar100
from keras.utils import to_categorical
import numpy as np
import sys
import h5py
### python3 model_avg.py weights/fitnet_DatasetAugment__Dropout_lr\:0.01_elu_1.h5 weights/fitnet_DatasetAugment__Dropout_lr\:0.01_elu_2.h5 weights/fitnet_DatasetAugment__Dropout_lr\:0.01_elu_3.h5 weights/fitnet_DatasetAugment__Dropout_lr\:0.01_elu_4.h5

# Computing the predictions of weights/fitnet_DatasetAugment__Dropout_lr:0.01_elu_1.h5
# Model 0 : Loss=1.28745703125, Accuracy=0.7008
# Computing the predictions of weights/fitnet_DatasetAugment__Dropout_lr:0.01_elu_2.h5
# Model 1 : Loss=1.2647326171875, Accuracy=0.706
# Computing the predictions of weights/fitnet_DatasetAugment__Dropout_lr:0.01_elu_3.h5
# Model 2 : Loss=1.28125048828125, Accuracy=0.7064
# Computing the predictions of weights/fitnet_DatasetAugment__Dropout_lr:0.01_elu_4.h5
# Model 3 : Loss=1.27666083984375, Accuracy=0.7036
# (10000, 100)
# Avg Model : Loss=0.9127634765625, Accuracy=0.7536

## ls weights/*.h5 | tr "\n" " " | xargs python3 model_avg.py
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
    with h5py.File(f, 'a') as fh5:
        if 'optimizer_weights' in fh5.keys():
            del fh5['optimizer_weights']
    
    model = load_model(f)

    print("Computing the predictions of {}".format(f))

    pred = model.predict(x_test,verbose=0)

    scores = acc_loss(pred, y_test)
    print("Model {} : Loss={}, Accuracy={}".format(idx, scores[0], scores[1]))
    if scores[0] <= 1.4):
        predictions.append(pred)
    del model
    
predictions = np.array(predictions)
pred_avg = np.mean(predictions, axis=0)
print(pred_avg.shape)
scores = acc_loss(pred_avg, y_test)
print("Avg Model : Loss={}, Accuracy={}".format(scores[0], scores[1]))
