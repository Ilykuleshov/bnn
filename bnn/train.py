from typing import *
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch import nn
from torch.optim import Optimizer
from tqdm import trange

from .bayes_base_module import BayesBaseModule
from .slgd import SLGD
from torch.utils.tensorboard import SummaryWriter

class BayesModel:
    def __init__(self, 
                 train_dataset: Dataset, 
                 test_dataset: Dataset, 
                 batch_size: int, 
                 lr: float, 
                 architecture,
                 device='cuda'):
        self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
        self.architecture = architecture.to(device)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = SLGD(self.architecture.parameters(), lr)
        self.device = device
    
    def fit(self, n_epochs):
        writer = SummaryWriter()
        for i in trange(n_epochs):
            loss = self.training_step()
            writer.add_scalar('Loss/train', loss, i)
            if n_epochs % 15 == 0:
                self.architecture.eval()
                with torch.no_grad():
                    for (x, y) in iter(self.test_dataloader):
                        x = x.to(self.device)
                        y = y.to(self.device)

                        y_pred = self.architecture(x)
                        loss_test = self.cross_entropy(y_pred, y) - self.architecture.log_prior()    
                    writer.add_scalar('Loss/test', loss_test / len(self.test_dataloader), i)
                self.architecture.train()
                        

    def training_step(self):
        for (x, y) in iter(self.train_dataloader):
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.architecture(x)
            loss = self.cross_entropy(y_pred, y) - self.architecture.log_prior()
            
            loss.backward()
            self.optimizer.step()
            
        return loss / len(self.train_dataloader)
