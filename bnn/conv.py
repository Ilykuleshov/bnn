import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from .bayes_base_module import BayesBaseModule
from .utils import init_with_distribution_

class BayesConv2d(nn.Conv2d, BayesBaseModule):
    r"""
    Applies Bayesian Convolution for 2D inputs

    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.

    .. note:: other arguments are following conv of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
    
    """
    def __init__(self, 
                 weight_distribution, 
                 bias_distribution, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 padding_mode='zeros', 
                 groups=1):
        
        super(BayesConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode)
        
        self.weight_distribution = weight_distribution
        self.bias_distribution = bias_distribution

#         init_with_distribution_(self.weight, self.weight_distribution)
#         init_with_distribution_(self.bias, self.bias_distribution)