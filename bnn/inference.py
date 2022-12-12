from typing import *
from pathlib import Path
import torch

from torch.distributions import Laplace, Normal
from torch import nn
from tqdm import trange
from statistics import mean, stdev

from .models import CNN, FCNN
from .train import BayesModel
from .utils import trainer_default_kwargs

def name_to_params(name: str):
    if name.startswith('long'):
        name = name[len('long_'):]

    model_name, distribution_name, temperature = name.split('_')
    temperature = float(temperature)

    model = {
        'cnn': CNN,
        'linear': FCNN,
        'fcnn': FCNN
    }[model_name.lower()]

    distribution = {
        'laplace': Laplace,
        'normal': Normal
    }[distribution_name.lower()](0, 0.05)

    return {'temperature': temperature, 'model': model, 'distribution': distribution}


def score_model(model_folder: Path, num_steps: int, **override_kwargs):
    params = name_to_params(model_folder.name)
    
    temperature = params['temperature']
    distribution = params['distribution']
    model: nn.Module = params['model'](weight_distribution=distribution, bias_distribution=distribution)

    model.load_state_dict(torch.load(model_folder / 'model.pth'))

    trainer_kwargs = trainer_default_kwargs()
    trainer_kwargs.update(override_kwargs)

    trainer = BayesModel(architecture=model, temperature=temperature, **trainer_kwargs)

    avg_state_dict = model.state_dict()
    for i in trange(num_steps):
        trainer.training_step()
        avg_state_dict = {k: v + model.state_dict()[k] for k, v in avg_state_dict.items()}

    avg_state_dict = {k: v / num_steps for k, v in avg_state_dict.items()}

    model.load_state_dict(avg_state_dict)
    return trainer.evaluate()['error']


def avg_over_runs(func: Callable, num_runs: int, *args, **kwargs):
    runs = [func(*args, **kwargs) for _ in trange(num_runs)]
    return mean(runs), stdev(runs)