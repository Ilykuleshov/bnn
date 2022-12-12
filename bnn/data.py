from torchvision.datasets import MNIST
import torchvision
import torch
from torch.utils.data import random_split

def get_data(seed=42):
    dataset = torchvision.datasets.MNIST('../notebooks/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

    generator=torch.Generator()
    generator.manual_seed(42)
    train_set, val_set = random_split(dataset, [50000, 10000], generator=generator)
    return train_set, val_set