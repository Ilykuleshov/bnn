

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from tqdm import trange
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_0 = nn.Linear(in_features = 28*28, out_features=100)
        self.fc_1 = nn.Linear(in_features = 100, out_features=100)
        self.fc_2 = nn.Linear(in_features = 100, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc_0(x))
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x
        
mdl = Model()

dataset = torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

train_dataloader = DataLoader(train_set, batch_size = 128, shuffle=True)
test_dataloader = DataLoader(val_set, batch_size = 1000, shuffle=True)

cross_entropy = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mdl.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 600, eta_min=0, last_epoch=- 1, verbose=False)

n_epochs = 150
device = 'cuda'

log_dir = '/homes/abazarova/bnn/notebooks/runs/sgd_fcnn_150ep'
writer = SummaryWriter(log_dir=log_dir)
mdl = mdl.to(device)
for i in trange(n_epochs):
    total_loss = 0
    for (x, y) in iter(train_dataloader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_pred = mdl(x)
        loss = cross_entropy(y_pred, y)
        total_loss += loss
        loss.backward()
        optimizer.step()
        scheduler.step()
    writer.add_scalar('Loss/train', total_loss / (len(train_dataloader)), i)
    if i % 15 == 0:
        mdl.eval()
        with torch.no_grad():
            total_loss = 0
            accuracy = 0
            for (x, y) in iter(test_dataloader):
                x = x.to(device)
                y = y.to(device)
                y_pred = mdl(x)
                loss_test = cross_entropy(y_pred, y)   
                total_loss += loss_test
                accuracy += (torch.argmax(y_pred, dim = -1) == y).sum() / len(y)
            writer.add_scalar('Loss/test', total_loss / len(test_dataloader), i)
            writer.add_scalar('Accuracy/test', accuracy / len(test_dataloader), i)
        mdl.train()
        
model_save_path = Path('/homes/abazarova/bnn/notebooks/models/fcnn_sgd/')
model_save_path.mkdir(exist_ok = True, parents = True)
torch.save(mdl.state_dict(), model_save_path / 'model.pth')