'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

# class Binarizer(torch.autograd.Function):
#     """Binarizes {0, 1} a real valued tensor."""

#     @staticmethod
#     def forward(ctx, inputs, threshold):
#         outputs = inputs.clone()
#         outputs[inputs.le(threshold)] = 0
#         outputs[inputs.gt(threshold)] = 1
#         return outputs

#     @staticmethod
#     def backward(ctx, grad_out):
#         return grad_out, None

class tsReLU(nn.Module):
    """docstring for tsReLU"""
    def __init__(self, in_features):
        super(tsReLU, self).__init__()
        self.in_features = in_features
        self.w = nn.Parameter(torch.Tensor(in_features).uniform_(0.01, 0.01))
        self.d = nn.Parameter(torch.Tensor([-0.01]))
        self.gate = nn.Sigmoid()

    def forward(self, x, s):
        if len(x.shape) == 2:
            shape = (1, -1)
        else:
            shape = (1, -1, 1, 1)

        x1 = self.gate(s*self.w.view(shape)) * x
        x1[x<0] *= self.gate(s*self.d)
        return x1
        # if self.d >= 0.5:
        #     return (self.w.view(shape)) * x
        # else:
        #     return (self.w.view(shape)) * F.relu(x)

    def binary_reg(self):
        return (self.w*(1-self.w)).sum() + 0.1*self.d*(1-self.d)

    def complexity_reg(self, s):
        return self.gate(s*self.w).sum() - 0.1*self.gate(s*self.d)

# class tsReLU(nn.Module):
#     """docstring for tsReLU"""
#     def __init__(self, in_features):
#         super(tsReLU, self).__init__()
#         self.in_features = in_features
#         self.w = nn.Parameter(torch.Tensor(in_features).uniform_(1.0, 1.0))
#         self.d = nn.Parameter(torch.Tensor([0.0]))
#         self.gate = Binarizer.apply

#     def forward(self, x, s):
#         if len(x.shape) == 2:
#             shape = (1, -1)
#         else:
#             shape = (1, -1, 1, 1)

#         x1 = self.gate(self.w, 0.5).view(shape) * x
#         x1[x<0] *= self.gate(self.d, 0.5)
#         return x1
#         # if self.d >= 0.5:
#         #     return (self.w.view(shape)) * x
#         # else:
#         #     return (self.w.view(shape)) * F.relu(x)

#     def binary_reg(self):
#         return (self.w*(1-self.w)).sum() + 0.1*self.d*(1-self.d)

#     def complexity_reg(self, s):
#         return self.gate(s*self.w).sum() - 0.1*self.gate(s*self.d)


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
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.AvgPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(512, 512),
            tsReLU(512),
            nn.Dropout(),
            nn.Linear(512, 512),
            tsReLU(512),
            nn.Linear(512, output_dim),
        ])
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

        self.PM = [m for m in self.modules() if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, tsReLU)]


    def forward(self, x, s):
        for m in self.layers:
            if isinstance(m, tsReLU):
                x = m(x, s)
            else:
                x = m(x)

        return x


def make_layers(cfg, n_channels, batch_norm=False):
    layers = []
    in_channels = n_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), tsReLU(v)]
            else:
                layers += [conv2d, tsReLU(v)]
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
            tsReLU(6),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            tsReLU(16),
            nn.AvgPool2d(kernel_size=2),
        ])

        self.smid = size
        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.AvgPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding)


        self.layers += nn.ModuleList([
            nn.Flatten(),
            nn.Linear(in_features=16*self.smid*self.smid, out_features=120),
            tsReLU(120),
            nn.Linear(in_features=120, out_features=84),
            tsReLU(84),
            nn.Linear(in_features=84, out_features=output_dim),
        ])

        self.PM = [m for m in self.modules() if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, tsReLU)]


    def forward(self, x, s):
        for m in self.layers:
            if isinstance(m, tsReLU):
                x = m(x, s)
            else:
                x = m(x)
        return x
