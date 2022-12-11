import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from bnn import BayesBaseModule, BayesConv2d, BayesLinear, BayesModel

class CNN(nn.Module):
    def __init__(self, 
                 weight_distribution,
                 bias_distribution):
        super().__init__()
        self.conv_0 = BayesConv2d(weight_distribution, bias_distribution, 
                                  in_channels=1, out_channels=64, kernel_size=3)
        self.conv_1 = BayesConv2d(weight_distribution, bias_distribution, 
                                  in_channels=64, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.fc = BayesLinear(weight_distribution, bias_distribution, in_features=1600, out_features=10)
        self._weight_distribution = weight_distribution
        self._bias_distribution = bias_distribution
        
    def weight_distribution(self):
        return self._weight_distribution
    
    def bias_distribution(self):
        return self._bias_distribution
    
    def weight(self):
        return self.weight
        
    def log_prior(self):
        log_p = 0
        for m in self.modules():
            if isinstance(m, (BayesLinear, BayesConv2d)):
                log_p += m.log_prior()
                
        return log_p

    def forward(self, x):
        x = self.relu(self.conv_0(x))
        x = self.pooling(x)
        x = self.relu(self.conv_1(x))
        x = self.pooling(x)
        x = x.view(-1, 1600)
        x = self.fc(x)
        return x
    
    
    
class FCNN(nn.Module):
    def __init__(self, 
                 weight_distribution,
                 bias_distribution):
        super().__init__()
        self.fc_0 = BayesLinear(weight_distribution, bias_distribution, 
                                  in_features = 28*28, out_features=100)
        self.fc_1 = BayesLinear(weight_distribution, bias_distribution, 
                                  in_features = 100, out_features=100)
        self.fc_2 = BayesLinear(weight_distribution, bias_distribution, 
                                  in_features = 100, out_features=10)
        self.relu = nn.ReLU()
        
        self._weight_distribution = weight_distribution
        self._bias_distribution = bias_distribution
        
    def weight_distribution(self):
        return self._weight_distribution
    
    def bias_distribution(self):
        return self._bias_distribution
    
    def weight(self):
        return self.weight
        
    def log_prior(self):
        log_p = 0
        for m in self.modules():
            if isinstance(m, (BayesLinear, BayesConv2d)):
                log_p += m.log_prior()
                
        return log_p

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc_0(x))
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x