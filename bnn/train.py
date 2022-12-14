from typing import *
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch import nn
from torch.optim import Optimizer
from tqdm import trange

from .bayes_base_module import BayesBaseModule
from .slgd import SLGD
from .utils import add_scalars
from torch.utils.tensorboard import SummaryWriter

class BayesModel:
    def __init__(self, 
                 train_dataset: Dataset, 
                 test_dataset: Dataset, 
                 batch_size: int, 
                 lr: float, 
                 temperature: float,
                 architecture: nn.Module,
                 device='cuda'):
        self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
        self.architecture = architecture.to(device)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.lr = lr
        self.num_data = len(train_dataset)
        self.optimizer = SLGD(self.architecture.parameters(), lr, temperature=temperature, num_data=self.num_data)
        self.device = device
    
    def fit(self, n_epochs, log_dir=None):
        writer = SummaryWriter(log_dir=log_dir)
        for i in trange(n_epochs):
            step_results = self.training_step()
            add_scalars(writer, 'Train', step_results, i)

            if i % 15 == 0:
                step_results = self.evaluate()
                add_scalars(writer, 'Test', step_results, i)

    def calc_losses(self, y_true, y_pred):
        ce_loss = self.cross_entropy(y_pred, y_true)
        prior_loss = - self.architecture.log_prior()
        accuracy = (torch.argmax(y_pred, dim = -1) == y_true).sum() / len(y_true)

        return ce_loss, prior_loss, accuracy
    
    def training_step(self):
        self.architecture.train()
        return self.step(self.train_dataloader)

    @torch.no_grad()
    def evaluate(self):
        self.architecture.eval()
        return self.step(self.test_dataloader)

    def step(self, dataloader):
        ce_loss_total = 0
        prior_loss_total = 0
        accuracy_total = 0

        for (x, y) in iter(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.architecture(x)
            ce_loss, prior_loss, accuracy = self.calc_losses(y, y_pred)
            loss: torch.Tensor = ce_loss + prior_loss
            
            if self.architecture.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            ce_loss_total += ce_loss.detach().item()
            prior_loss_total += prior_loss.detach().item()
            accuracy_total += accuracy.detach().item()
            
        ce_loss_total /= len(dataloader)
        prior_loss_total /= len(dataloader)
        accuracy_total /= len(dataloader)

        return {'ce_loss': ce_loss_total, 'prior_loss': prior_loss_total, 'error': 1 - accuracy_total}