import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from tqdm import trange
from torch.distributions.normal import Normal
from bnn import BayesBaseModule, BayesConv2d, BayesLinear, BayesModel, CNN
from torch.distributions.laplace import Laplace
from pathlib import Path

distr = Normal(torch.tensor(0.), torch.tensor(0.05))
mdl = CNN(weight_distribution = distr,
                 bias_distribution = distr)
                 
dataset = torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

temperatures = [1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
errors = np.zeros(len(temperatures))
distr = Normal(torch.tensor(0.), torch.tensor(0.05))
n_epochs = 1000

for i, temperature in enumerate(temperatures[4:]):
    mdl = CNN(weight_distribution = distr, bias_distribution = distr)
    trainer = BayesModel(train_dataset=train_set,
                    test_dataset=val_set,
                    batch_size=128,
                    architecture=mdl,
                    lr=1e-3,
                    temperature=temperature)
    trainer.fit(n_epochs = 1000, log_dir='./runs/long_cnn_normal_{temp:.3f}'.format(temp=temperature))
    model_save_path = Path('./models/long_cnn_normal_{temp:.3f}/'.format(temp=temperature))
    model_save_path.mkdir(exist_ok = True, parents = True)
    torch.save(trainer.architecture.state_dict(), model_save_path / 'model.pth')