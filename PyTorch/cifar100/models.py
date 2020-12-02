import torch.nn as nn
import torch.nn.functional as F

################################################# Model

# Architecture
# Transfer function : ELU
# L2 regularization of 0.0005 in all the conv layers
# Convolution blocks : Conv - BN - ELU
# C3s1 x 32, C3s1 x 64, Max2s2, C3s1x128, C3s1x256, GlobalAvg, Dense(500), Dropout(0.5), Dense(100), Softmax

def conv_BN_relu(cin, cout, with_BN):
    batchnorm_momentum = 0.99
    batchnorm_epsilon = 1e-3

    layers = [
        nn.Conv2d(cin, cout, 3, padding=1, bias=not with_BN),
        nn.Conv2d(cout, cout, 3, padding=1, bias=not with_BN)
    ]
    if with_BN:
        layers.append(nn.BatchNorm2d(32,
                                     eps=batchnorm_epsilon,
                                     momentum=batchnorm_momentum))
    layers.append(nn.ELU(inplace=True))
    return layers


class Net(nn.Module):

    def __init__(self, use_dropout, use_batchnorm, use_l2reg):
        super(Net, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.use_l2reg = use_l2reg

        # We disable the bias if we use the batchnorm as the BN
        # already captures the bias
        self.use_bias = not self.use_batchnorm

        self.model = nn.Sequential(
            *conv_BN_relu(3, 32, use_batchnorm),
            *conv_BN_relu(32, 32, use_batchnorm),
            nn.MaxPool2d(2),
            *conv_BN_relu(32, 64, use_batchnorm),
            *conv_BN_relu(64, 64, use_batchnorm),
            nn.MaxPool2d(2),
            *conv_BN_relu(64, 128, use_batchnorm),
            *conv_BN_relu(128, 128, use_batchnorm),
            nn.MaxPool2d(2),
            *conv_BN_relu(128, 256, use_batchnorm),
            *conv_BN_relu(256, 256, use_batchnorm),
            nn.AvgPool2d(4)
        )

        self.fc1   = nn.Linear(256, 500)
        self.elu1  = nn.ELU(inplace=True)
        if self.use_dropout:
            self.drop  = nn.Dropout2d(0.5)
        self.fc2   = nn.Linear(500, 100)

        if use_l2reg:
            pass
        
        self.init()
        
    def init(self):
        for m in self.modules():
            if   isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if self.use_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):

        features = self.model(x)
        features = features.view(-1, self.num_flat_features(features))

        x = self.fc1(features)
        x = self.elu1(x)

        if self.use_dropout:
            x = self.drop(x)

        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



