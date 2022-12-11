from typing import *

from torch import nn
from torch import distributions as td
import torch

from .utils import init_with_distribution_
from .bayes_base_module import BayesBaseModule

class BayesLinear(nn.Linear, BayesBaseModule):
    def __init__(self, in_features, out_features, weight_distribution: td.Distribution, bias_distribution: Union[td.Distribution, None]=None, **kwargs) -> None:
        super().__init__(in_features=in_features, out_features=out_features, **kwargs)
        self.weight_distribution = weight_distribution
        self.bias_distribution = bias_distribution

       # init_with_distribution_(self.weight, self.weight_distribution)
      #  init_with_distribution_(self.bias, self.bias_distribution)
