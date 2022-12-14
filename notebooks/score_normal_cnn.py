from bnn.inference import score_model, avg_over_runs
from bnn.utils import trainer_default_kwargs
from pathlib import Path
from tqdm import trange
import json

temperatures = [1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
errors = []
stdevs = []

kwargs = trainer_default_kwargs()
kwargs['lr'] = 1e-9
num_repeat = 10
num_steps = 10

for t in temperatures:
    model_path = Path('/homes/abazarova/bnn/notebooks/models/long_cnn_normal_{temp:.3f}/'.format(temp=t))

    error, stdev = avg_over_runs(score_model, num_repeat, model_path, num_steps, **kwargs)
    print(f't={t} => {error:.4f}+-{stdev:.5f}')
    errors.append(error)
    stdevs.append(stdev)

results = {}
results['cnn'] = (errors, stdevs)

with open('./normal_results_cnn_OLDLOSS.json', 'w') as file:
    json.dump(results, file)