from typing import *

from torch import nn
from torch import distributions as td
import torch

from .utils import init_distributed_
from .bayes_base_module import BayesBaseModule

class BayesLinear(nn.Linear, BayesBaseModule):
    def __init__(self, weight_distribution: td.Distribution, bias_distribution: Union[td.Distribution, None]=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weight_distribution = weight_distribution
        self.bias_distribution = bias_distribution

        init_distributed_(self.weight, self.weight_distribution)
        init_distributed_(self.bias, self.bias_distribution)
