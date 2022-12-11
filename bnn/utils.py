from typing import *

import torch
from torch.nn.parameter import Parameter
from torch.distributions import Distribution

@torch.no_grad()
def init_distributed_(tensor: Union[torch.Tensor, None], distribution: Distribution):
    if tensor is not None:
        sample = distribution.sample(tensor.shape)
        assert sample.shape == tensor.shape, f'Shape mismatch! {sample.shape} != {tensor.shape}'
        tensor.set_(sample)
        return tensor