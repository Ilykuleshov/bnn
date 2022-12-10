from typing import *
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch import nn
from torch.optim import Optimizer

class BayesModel:
    def __init__(self, dataset: Dataset, batch_size: int, architecture: nn.Module, optimizer: Optimizer):
        self.dataloader = DataLoader(dataset, batch_size, shuffle=True)
        self.architecture = architecture
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def fit(self, n_epochs):
        for i in range(n_epochs):
            self.training_step()

    def traininig_step(self):
        for (x, y) in iter(self.dataloader):
            x = x.to(self.architecture)
            y = y.to(self.architecture)

            y_pred = self.architecture(x)
            loss += self.nnl()