
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
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers
import glob
import h5py
import numpy as np 

#datapath_train = os.path.join(*["data","sample","train"])
#datapath_valid = os.path.join(*["data","sample", "valid"])

datapath_train = os.path.join(*["data","train"])
datapath_valid = os.path.join(*["data","valid"])

datapath_test = os.path.join(*["data","test"])

nb_training_data = len(glob.glob(os.path.join(datapath_train, "*/*.jpg")))
nb_validation_data = len(glob.glob(os.path.join(datapath_valid, "*/*.jpg")))
nb_testing_data = len(glob.glob(os.path.join(datapath_test, "*/*.jpg")))
img_size = (224, 224)
batch_size = 16
nb_epochs = 20

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
def vgg_preprocess(x):
	x = x - vgg_mean
	return x[:, ::-1] # reverse axis rgb->bgr

#train_datagen = ImageDataGenerator(featurewise_std_normalization=True)
# Requires to be fitted to the data, and may need a fit_from_directory function. It also depends on what the pretrained networks expect from their input. 
train_datagen = ImageDataGenerator(rescale=1./255., preprocessing_function=vgg_preprocess)
valid_datagen = ImageDataGenerator(rescale=1./255., preprocessing_function=vgg_preprocess)

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
                                                    target_size=img_size,
						    shuffle=False)
test_generator = valid_datagen.flow_from_directory(datapath_test,
                                                   batch_size=1,
                                                   target_size=img_size,
						   shuffle=False)


# Get our pretrained model
loaded_model = VGG16(input_shape=img_size+(3,),
                     include_top=True, weights='imagenet')
# Disable training of the layers up to the last FC bottleneck features
# i.e. we allow training only of the last FC layers
for layer in loaded_model.layers[:-3]:
        layer.trainable = False
# Cut off the head and stack a bi-class classification layer
loaded_model.layers.pop() # Remove the last classification layer
last = loaded_model.layers[-1].output
preds = Dense(1, activation='sigmoid')(last)
model = Model(input=loaded_model.input, output=preds)
model.summary()

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Callbacks
checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)


# Fit the pretrained model
print("Training")
model.fit_generator(train_generator,
                    steps_per_epoch=nb_training_data//batch_size,
                    epochs=nb_epochs,
                    validation_data=valid_generator, validation_steps=nb_validation_data//batch_size,
                    verbose=1,
		    callbacks=[checkpoint_cb])

# Generate the class probabilities for all the test images

with h5py.File("best_model.h5", 'a') as f:
    if 'optimizer_weights' in f.keys():
	    del f['optimizer_weights']

model = load_model("best_model.h5")

print("Testing")
pred = model.predict_generator(test_generator,
		               steps=nb_testing_data).ravel().tolist()

# We fill in a submission array with the results
submission = np.zeros((nb_testing_data, 2))
submission[:,0] = list(map(lambda fname: int(os.path.basename(fname).split('.')[0]), test_generator.filenames))
submission[:,1] = pred

# That we need to sort by image id
submission = submission[submission[:,0].argsort()]

fh = open('submission.csv','w')
fh.write('id,label\n')
for l in submission:
	fh.write('%i,%f\n'%(l[0], l[1]))
        fh.close()
