from typing import *

import torch
from torch.distributions import Distribution

@torch.no_grad()
def init_with_distribution_(tensor: Union[torch.Tensor, None], distribution: Distribution):
    if tensor:
        tensor.set_(distribution.sample(tensor.shape))

