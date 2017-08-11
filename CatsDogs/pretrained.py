
# This script loads a model pretrained on ImageNet
# Cuts off the head and replace it with 2 outputs
# in order to classify Cats and Dogs

# See also : https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# It accepts few options :

# --fine_tune=<all, top>   : Fine tune only the last Dense layer or the whole architecture
#

import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.layers import Dropout
import glob

datapath_train = os.path.join(*["data","sample","train"])
datapath_valid = os.path.join(*["data","sample", "valid"])


#datapath_train = os.path.join(*["data","train"])
#datapath_valid = os.path.join(*["data","valid"])
datapath_test = os.path.join(*["data","test"])

nb_training_data = len(glob.glob(os.path.join(datapath_train, "*/*.jpg")))
nb_validation_data = len(glob.glob(os.path.join(datapath_valid, "*/*.jpg")))
nb_test_data = len(glob.glob(os.path.join(datapath_test, "*.jpg")))
img_size = (224, 224)
batch_size=32

#train_datagen = ImageDataGenerator(featurewise_std_normalization=True)
# Requires to be fitted to the data, and may need a fit_from_directory function. It also depends on what the pretrained networks expect from their input. 
train_datagen = ImageDataGenerator(rescale=1./255.)
valid_datagen = ImageDataGenerator(rescale=1./255.)

# Testing the generator
# img = load_img("data/sample/train/cat/0.jpg")
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)
# b = datagen.flow(x, batch_size=1)
# tx = next(b)
# import matplotlib.pyplot as plt
# plt.imshow(tx[0,:,:,:])
# plt.colorbar()
# plt.show()

train_generator = train_datagen.flow_from_directory(datapath_train,
                                                    class_mode="binary",
                                                    batch_size=batch_size,
                                                    target_size=img_size)
valid_generator = valid_datagen.flow_from_directory(datapath_valid,
                                                    class_mode='binary',
                                                    batch_size=batch_size,
                                                    target_size=img_size)

# Get our pretrained model
loaded_model = VGG16(input_shape=img_size+(3,),
                     include_top=True, weights='imagenet')
for layer in loaded_model.layers:
    layer.trainable = False
# Cut off the head and stack a bi-class classification layer
#flat = Flatten()(loaded_model.output)
#drop = Dropout(0.3)(flat)
loaded_model.layers.pop() # Remove the last classification layer
last = loaded_model.layers[-1].output
preds = Dense(1, activation='sigmoid')(last)
model = Model(input=loaded_model.input, output=preds)
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the pretrained model
model.fit_generator(train_generator,
                    steps_per_epoch=nb_training_data//batch_size,
                    epochs=20,
                    validation_data=valid_generator, validation_steps=nb_validation_data//batch_size,
                    verbose=1)

# Generate the class probabilities for all the test images
# model.predict_generator()

fh = open('submission.csv','w')
fh.write('id,label')

fh.close()
