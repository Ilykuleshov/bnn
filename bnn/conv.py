import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.distributions.multivariate_normal import MultivariateNormal

class BayesConv2d(nn.Conv2d):
    r"""
    Applies Bayesian Convolution for 2D inputs

    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.

    .. note:: other arguments are following conv of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
    
    """
    def __init__(self, 
                 prior_mu, 
                 prior_sigma, 
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
        
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        m = MultivariateNormal(torch.ones(self.weight.numel()) * prior_mu,
                              torch.eye(self.weight.numel()) * prior_sigma)
        sampled_weights = m.rsample()
        self.weight = nn.Parameter(sampled_weights.reshape(self.weight.shape))