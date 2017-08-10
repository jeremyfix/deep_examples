
# This script loads a model pretrained on ImageNet
# Cuts off the head and replace it with 2 outputs
# in order to classify Cats and Dogs

# See also : https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# It accepts few options :

# --fine_tune=<all, top>   : Fine tune only the last Dense layer or the whole architecture
#

import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

datapath_train = os.path.join(*["data","sample","train"])
datapath_valid = os.path.join(*["data","sample", "valid"])

#datapath_train = os.path.join(*["data","train"])
#datapath_valid = os.path.join(*["data","valid"])

img_size = (224, 224)

#train_datagen = ImageDataGenerator(featurewise_std_normalization=True)
# Requires to be fitted to the data, and may need a fit_from_directory function
datagen = ImageDataGenerator(rescale=1./255., rotation_range=40)

img = load_img("data/sample/train/cat/0.jpg")
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)
b = datagen.flow(x, batch_size=1)
tx = next(b)
import matplotlib.pyplot as plt
plt.imshow(tx[0,:,:,:])
plt.colorbar()
plt.show()

train_generator = datagen.flow_from_directory(datapath_train,
                                              class_mode="binary",
                                              batch_size=32,
                                              target_size=img_size)
