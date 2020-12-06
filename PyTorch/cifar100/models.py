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

def conv_BN_relu(cin, cout, ksize, padding, with_BN, tf_function, stride=1):

    layers = [
        nn.Conv2d(cin, cout,
                  ksize, padding=padding, stride=stride,
                  bias=not with_BN)
    ]
    if with_BN:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(tf_function())
    return layers

def convBlock(cin, cout, with_BN, tf_function):
    cinter = int(math.sqrt(cout/cin)) * cin

    convbnrelu = functools.partial(conv_BN_relu,
                                   with_BN=with_BN,
                                   tf_function=tf_function)

    layers = convbnrelu(cin, cinter, (1, 3), (0, 1)) + \
             convbnrelu(cinter, cout, (3, 1), (1, 0)) + \
             convbnrelu(cout, cout, (1,3), (0, 1)) + \
             convbnrelu(cout, cout, (3, 1), (1, 0))
    return layers

class Net(nn.Module):

    def __init__(self, use_dropout, use_batchnorm, use_l2reg):
        super(Net, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.use_bias = not self.use_batchnorm
        self.use_l2reg = use_l2reg
        if use_l2reg:
            self.l2_reg = 0.001
        else:
            self.l2_reg = 0

        tf_function = lambda: nn.ReLU(inplace=True)

        # The RF size is 32x32
        self.model = nn.Sequential(
            *convBlock(3, 32, use_batchnorm, tf_function),
            *convBlock(32, 32, use_batchnorm, tf_function),
            *conv_BN_relu(32, 32,
                          (3,3) , (1,1),
                          self.use_batchnorm, tf_function, stride=2),
            # nn.MaxPool2d(2)
            *convBlock(32, 128, use_batchnorm, tf_function),
            *convBlock(128, 128, use_batchnorm, tf_function),
            *conv_BN_relu(128, 128,
                          (3,3) , (1,1),
                          self.use_batchnorm, tf_function, stride=2),
            # nn.MaxPool2d(2),
            *convBlock(128, 512, use_batchnorm, tf_function),
            *convBlock(512, 512, use_batchnorm, tf_function),
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                regularization += m.weight.norm(2)
            elif isinstance(m, nn.Linear):
                regularization += m.weight.norm(2)
        return self.l2_reg * regularization

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
