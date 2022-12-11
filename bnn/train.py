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
                 temperature: float,
                 architecture,
                 device='cuda'):
        self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
        self.architecture = architecture.to(device)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = SLGD(self.architecture.parameters(), lr, temperature=temperature)
        self.eff_num_data = len(train_dataset)
        self.device = device
    
    def fit(self, n_epochs):
        writer = SummaryWriter()
        for i in trange(n_epochs):
            loss = self.training_step()
            writer.add_scalar('Loss/train', loss, i)
            if i % 15 == 0:
                self.architecture.eval()
                with torch.no_grad():
                    total_loss = 0
                    accuracy = 0
                    for (x, y) in iter(self.test_dataloader):
                        x = x.to(self.device)
                        y = y.to(self.device)

                        y_pred = self.architecture(x)
                        loss_test = self.cross_entropy(y_pred, y) - self.architecture.log_prior()    
                        total_loss += loss_test
                        accuracy += (torch.argmax(y_pred, dim = -1) == y).sum() / len(y)
                    writer.add_scalar('Loss/test', total_loss / len(self.test_dataloader), i)
                    writer.add_scalar('Accuracy/test', accuracy / len(self.test_dataloader), i)
                self.architecture.train()
                                      
    def training_step(self):
        total_loss = 0
        for (x, y) in iter(self.train_dataloader):
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.architecture(x)
            loss = self.cross_entropy(y_pred, y) - self.architecture.log_prior() / self.eff_num_data
            total_loss += loss
            loss.backward()
            self.optimizer.step()
            
        return total_loss / len(self.train_dataloader)
    
    def evaluate(self):
        with torch.no_grad():
            accuracy = 0
            for (x, y) in iter(self.test_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.architecture(x)
                accuracy += (torch.argmax(y_pred, dim = -1) == y).sum() / len(y)
        return accuracy / len(self.test_dataloader)
