'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Bernoulli, LogNormal, Normal
from utils import *


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class VBD_Layer(nn.Module):
    """docstring for ClassName"""
    def __init__(self, in_features):
        super(VBD_Layer, self).__init__()
        self.in_features = in_features
        self.mu = nn.Parameter(torch.Tensor(in_features).uniform_(1.0, 1.0))
        # self.mu = torch.ones(in_features).cuda()
        self.log_sigma2 = nn.Parameter(torch.Tensor(in_features).uniform_(-5.0, -5.0))
        self.normal = Normal(0, 1)
        self.thres = 3.0

    def forward(self, x, mask=True): 

        if len(x.shape) == 2:
            shape = (1, -1)
        else:
            shape = (1, -1, 1, 1)
        if self.training:
            sigma = torch.exp(0.5*self.log_sigma2)
            epsilon = self.normal.sample(self.log_sigma2.size()).cuda()
            x = x * (self.mu + sigma * epsilon).view(shape)
        else:
            x = x * self.mu.view(shape)
        # x = x * self.mu
        # if mask:
        x = x * (self.log_alpha < self.thres).view(shape)
        return x

    def kl_divergence(self):
        kld = 0.5*torch.log1p(self.mu*self.mu/(torch.exp(self.log_sigma2)+1e-8))
        return kld.sum()

    @property
    def log_alpha(self):
        return self.log_sigma2 - 2.0 * torch.log(self.mu + 1e-8)

    def get_mask(self):
        return (self.log_alpha < self.thres)



class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, input_dim, output_dim, cfg, batch_norm=False):
        super(VGG, self).__init__()
        n_channels, size, _ = input_dim
        self.layers = make_layers(cfg, n_channels, batch_norm=batch_norm)

        self.smid = size
        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            # nn.Dropout(),
            nn.Linear(512*self.smid*self.smid, 512),
            nn.ReLU(True),
            VBD_Layer(512),
            # nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            VBD_Layer(512),
            nn.Linear(512, output_dim),
        ])
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

        self.PM = [m for m in self.modules() if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, VBD_Layer)]


    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


def make_layers(cfg, n_channels, batch_norm=False):
    layers = []
    in_channels = n_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True), VBD_Layer(v)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True), VBD_Layer(v)]
            in_channels = v
    return nn.ModuleList(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(input_dim, output_dim):
    """VGG 11-layer model (configuration "A")"""
    return VGG(input_dim, output_dim, cfg['A'], batch_norm=False)


def vgg11_bn(input_dim, output_dim):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(input_dim, output_dim, cfg['A'], batch_norm=True)


def vgg13(input_dim, output_dim):
    """VGG 13-layer model (configuration "B")"""
    return VGG(input_dim, output_dim, cfg['B'], batch_norm=False)


def vgg13_bn(input_dim, output_dim):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(input_dim, output_dim, cfg['B'], batch_norm=True)


def vgg16(input_dim, output_dim):
    """VGG 16-layer model (configuration "D")"""
    return VGG(input_dim, output_dim, cfg['C'], batch_norm=False)


def vgg16_bn(input_dim, output_dim):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(input_dim, output_dim, cfg['C'], batch_norm=True)


def vgg19(input_dim, output_dim):
    """VGG 19-layer model (configuration "E")"""
    return VGG(input_dim, output_dim, cfg['D'], batch_norm=False)


def vgg19_bn(input_dim, output_dim):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(input_dim, output_dim, cfg['D'], batch_norm=True)


class LeNet5(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LeNet5, self).__init__()
        
        n_channels, size, _ = input_dim
        self.layers = nn.ModuleList([           
            nn.Conv2d(in_channels=n_channels, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            VBD_Layer(6),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            VBD_Layer(16),
        ])

        self.smid = size
        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)


        self.layers += nn.ModuleList([
            nn.Flatten(),
            nn.Linear(in_features=16*self.smid*self.smid, out_features=120),
            nn.ReLU(),
            VBD_Layer(120),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            VBD_Layer(84),
            nn.Linear(in_features=84, out_features=output_dim),
        ])

        self.PM = [m for m in self.modules() if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, VBD_Layer)]


    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x