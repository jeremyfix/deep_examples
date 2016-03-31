import numpy as np
import caffe

# 
caffe.set_mode_cpu()

# 

net = caffe.Net('classify_xor.prototxt', caffe.TEST
