import sys

from math import sqrt, ceil
import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if(len(sys.argv) != 2):
    print("Usage : {} model.hdf5".format(sys.argv[0]))
    sys.exit(-1)

model_fname = sys.argv[1]

model = keras.models.load_model(model_fname)
model.summary()


#### Visualize the convolution kernels of the first layer
l = model.get_layer('conv1')
w, b = l.get_weights()
# w is (w, h, c, num_kernel)
# for the first layer, on MNIST, c=1 
Nw = w.shape[-1]

Nrows = int(ceil(sqrt(Nw)))
Ncols = int(ceil(Nw / Nrows))
print(Nw, Nrows, Ncols)

plt.figure()
plt.title('Conv1 filters', fontsize=20)

for i in range(Nw):
    plt.subplot(Nrows, Ncols, i+1)
    plt.imshow(w[:,:,0,i], cmap='gray_r')

plt.show()


##### Plot some statistics
## For example, look if some cells at max2 predominently respond for
## samples of specific classes... but if that would be the case
## the problem would already be linearly separable.
# Check out the visualization of computing th einput for which max2 cells
# maximally respond..
