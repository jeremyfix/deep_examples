import matplotlib
import cv2
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

x0 = cv2.imread('MickeyArt.jpg')
b,g,r = cv2.split(x0)       # get b,g,r
x0 = cv2.merge([r,g,b])     # switch it to rgb

img_height, img_width = x0.shape[0], x0.shape[1]

datagen = ImageDataGenerator(width_shift_range=5./img_width,
                             height_shift_range=5./img_height,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             rotation_range=90)

gen = datagen.flow(np.expand_dims(x0, axis=0),
                   np.expand_dims([0], axis=0),
                   batch_size=1)

plt.figure()
plt.imshow(x0)
plt.axis('off')
plt.savefig('orig.pdf', bbox_inches='tight')

fig, axarr = plt.subplots(4,4)
fig.patch.set_facecolor('white')
for i in range(4):
    for j in range(4):
        X, y = next(gen)
        axarr[i,j].imshow(X[0,:,:,:].astype(np.uint8))
        axarr[i,j].axis('off')

#plt.tight_layout()
plt.savefig('augmented.pdf', bbox_inches='tight')

