from typing import *

import torch
from torch.nn.parameter import Parameter
from torch.distributions import Distribution
from torch.utils.tensorboard import SummaryWriter

from bnn.data import get_data

@torch.no_grad()
def init_distributed_(tensor: Union[torch.Tensor, None], distribution: Distribution):
    if tensor is not None:
        sample = distribution.sample(tensor.shape)
        assert sample.shape == tensor.shape, f'Shape mismatch! {sample.shape} != {tensor.shape}'
        tensor.set_(sample)
        return tensor


def add_scalars(writer: SummaryWriter, parent_tag: str, scalars: Dict[str, float], i: int):
    for k, v in scalars.items():
        writer.add_scalar(f'{parent_tag}/{k}', v, i)


def trainer_default_kwargs():
    train_set, val_set = get_data()

    return dict(
        batch_size=128,
        lr=1e-3,
        train_dataset = train_set,
        test_dataset = val_set,
        device='cuda'
    )
