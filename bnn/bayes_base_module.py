from abc import ABC, abstractproperty
from typing import *

from torch import distributions as td
from torch import nn
import torch

class BayesBaseModule(ABC, nn.Module):
    def log_prior(self) -> torch.Tensor:
        log_prob: torch.Tensor = self.weight_distribution.log_prob(self.weight).sum()
        if self.bias_distribution:
            log_prob = log_prob + self.bias_distribution.log_prob(self.bias).sum()

        return log_prob