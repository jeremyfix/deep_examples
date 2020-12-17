import math
import torch.nn as nn
import torch.nn.functional as F
import functools

################################################# Model

# Architecture
# Transfer function : ELU
# L2 regularization of 0.0005 in all the conv layers
# Convolution blocks : Conv - BN - ELU
# C3s1 x 32, C3s1 x 64, Max2s2, C3s1x128, C3s1x256, GlobalAvg, Dense(500), Dropout(0.5), Dense(100), Softmax

def BN_relu_conv(cin, cout, ksize, padding, with_BN, tf_function, stride=1):

    layers = []
    if with_BN:
        layers.append(nn.BatchNorm2d(cin))
    layers.append(tf_function())
    layers.append(
        nn.Conv2d(cin, cout,
                  ksize, padding=padding, stride=stride,
                  bias=not with_BN)
    )
    return layers

def conv_BN_relu(cin, cout, ksize, padding, with_BN, tf_function, stride=1):

    layers = []
    layers.append(
        nn.Conv2d(cin, cout,
                  ksize, padding=padding, stride=stride,
                  bias=not with_BN)
    )
    if with_BN:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(tf_function())
    return layers

def convBlock(cin, cout, with_BN, tf_function):
    cinter = int(math.sqrt(cout/cin)) * cin

    bnreluconv = functools.partial(BN_relu_conv,
                                   with_BN=with_BN,
                                   tf_function=tf_function)
    # convbnrelu = functools.partial(conv_BN_relu,
    #                                with_BN=with_BN,
    #                                tf_function=tf_function)

    layers = bnreluconv(cin, cinter, (1, 3), (0, 1)) + \
             bnreluconv(cinter, cout, (3, 1), (1, 0)) + \
             bnreluconv(cout, cout, (1,3), (0, 1)) + \
             bnreluconv(cout, cout, (3, 1), (1, 0))
    return layers

class Net(nn.Module):

    def __init__(self, use_dropout, use_batchnorm, l2reg):
        super(Net, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.use_bias = not self.use_batchnorm
        self.l2_reg = l2reg

        tf_function = lambda: nn.ReLU()

        # The RF size is 32x32
        self.model = nn.Sequential(
            *convBlock(3, 96, use_batchnorm, tf_function),
            *convBlock(96, 96, use_batchnorm, tf_function),
            *conv_BN_relu(96, 96,
                          (3,3) , (1,1),
                          self.use_batchnorm, tf_function, stride=2),
            # nn.MaxPool2d(2)
            *convBlock(96, 192, use_batchnorm, tf_function),
            *convBlock(192, 192, use_batchnorm, tf_function),
            *conv_BN_relu(192, 192,
                          (3,3) , (1,1),
                          self.use_batchnorm, tf_function, stride=2),
            *conv_BN_relu(192, 512, (1,1) , (0,0),
                          self.use_batchnorm, tf_function),
            *conv_BN_relu(512, 512, (1,1) , (0,0),
                          self.use_batchnorm, tf_function),
            *conv_BN_relu(512, 512, (1,1) , (0,0),
                          self.use_batchnorm, tf_function),
            # nn.MaxPool2d(2),
            # *convBlock(128, 512, use_batchnorm, tf_function),
            # *convBlock(512, 512, use_batchnorm, tf_function),
            nn.AvgPool2d(8)
        )
        
        classifier_layers = []
        if self.use_dropout:
            classifier_layers.append(nn.Dropout2d(0.5))

        classifier_layers.append(nn.Linear(512, 100))
        self.classifier = nn.Sequential(*classifier_layers) 

        self.init()
        
    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data,
                                       nonlinearity='relu')
                if self.use_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data,
                                       nonlinearity='relu')
                m.bias.data.zero_()
    
    def penalty(self):
        regularization = 0
        if self.l2reg is not None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    regularization += m.weight.norm(2)
                elif isinstance(m, nn.Linear):
                    regularization += m.weight.norm(2)
            regularization *= self.l2_reg
        return regularization

    def forward(self, x):
        features = self.model(x)
        features = features.view(-1, self.num_flat_features(features))
        return self.classifier(features)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
