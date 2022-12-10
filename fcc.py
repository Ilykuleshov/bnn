from typing import *

from torch import nn
from torch import distributions as td
import torch

class BayesLinear(nn.Linear):
    def __init__(self, weight_distribution: td.Distribution, bias_distribution: Union[td.Distribution, None]=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weight_distribution = weight_distribution
        self.bias_distribution = bias_distribution
        self.weight = nn.Parameter(torch.zeros((super().in_features, super().out_features)))#weight_distribution.sample((super().in_features, super().out_features))
        self.bias = bias_distribution.rsample(super().in_features) if bias_distribution else None
