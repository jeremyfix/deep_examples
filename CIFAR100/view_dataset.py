from keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
print("Loading the dataset")
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

print("I loaded {} training images of size {} x {} x {}".format(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3]))
print("I loaded {} test images of size {} x {} x {}".format(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3]))

print("Viewing 16 samples of the training set")
idx = np.arange(y_train.shape[0])
np.random.shuffle(idx)
idx = idx[16:]

fig, axarr = plt.subplots(4,4)
fig.patch.set_facecolor('white')
for i in range(4):
    for j in range(4):
        axarr[i,j].imshow(x_train[idx[i*4+j],:,:,:], interpolation='None')
        axarr[i,j].set_title(CIFAR100_LABELS_LIST[y_train[idx[i*4+j], 0]])
        axarr[i,j].axis('off')

plt.tight_layout()
plt.show()

