## We train a Conv - Relu - Conv - Relu ...
# network to fill in the missing part of the data
# where we intentionnaly mask x % of the pixels
# We reconstruct the unsmaked image or the residual
# The loss is a pixelwise L2 norm

import os
import glob
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import DirectoryIterator, Iterator

datapath_train = os.path.expanduser("~/Datasets/CatsDogs/data/train")


datagen = ImageDataGenerator(rescale=1./255.)

##### En fait, il suffit d'ajouter une fonction de corruption
##### dans le ImageDataGenerator
#####

class MyCorruptor():

    def __init__(self, generator, sparse_ratio):
        self._generator = generator

    def next(self):
        batch_x, batch_y = next(self._generator)
        # Apply the corruption on the input
        batch_x

batch_size = 32
img_size = (224, 224)
MyCorruptor(train_datagen.flow_from_directory(datapath_train,
                                             class_mode="input",
                                             batch_size=batch_size,
                                             target_size=img_size))
