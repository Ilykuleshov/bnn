from typing import *
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch import nn
from torch.optim import Optimizer

from .bayes_base_module import BayesBaseModule
from .slgd import SLGD


class BayesModel:
    def __init__(self, dataset: Dataset, batch_size: int, architecture: BayesBaseModule):
        self.dataloader = DataLoader(dataset, batch_size, shuffle=True)
        self.architecture = architecture
        self.cross_entropy = nn.CrossEntropyLoss()
        self.optimizer = SLGD(self.architecture.parameters())
    
    def fit(self, n_epochs):
        for i in range(n_epochs):
            self.training_step()

    def training_step(self):
        for (x, y) in iter(self.dataloader):
            self.optimizer.zero_grad()
            x = x.to(self.architecture)
            y = y.to(self.architecture)

            y_pred = self.architecture(x)
            loss = self.cross_entropy(y_pred, y) - self.architecture.log_prior()
            
            loss.backward()
            self.optimizer.step()
