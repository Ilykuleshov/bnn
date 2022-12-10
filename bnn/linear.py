from typing import *

from torch import nn
from torch import distributions as td
import torch

class BayesLinear(nn.Linear):
    def __init__(self, in_features, out_features, weight_distribution: td.Distribution, bias_distribution: Union[td.Distribution, None]=None, **kwargs) -> None:
        super().__init__(in_features=in_features, out_features=out_features, **kwargs)
        self.weight_distribution = weight_distribution
        self.bias_distribution = bias_distribution

        with torch.no_grad():
            self.weight.set_(weight_distribution.sample((in_features, out_features)))
            if bias_distribution:
                self.bias.set_(bias_distribution.sample(in_features))
