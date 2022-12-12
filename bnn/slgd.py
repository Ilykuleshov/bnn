import torch
from torch.optim import Optimizer
from math import sqrt


class SLGD(Optimizer):

    def __init__(self, params, lr, num_data, temperature: float = 1, *, maximize=False, ):
        defaults = dict(lr=lr, maximize=maximize, temperature=temperature, num_data=num_data)
        super(SLGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('temperature', 1)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            lr = group['lr']
            maximize = group['maximize']
            temperature = group['temperature']
            num_data = group['num_data']

            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]

                # add tempered noise
                if temperature > 0:
                    d_p = d_p.add(torch.randn_like(d_p).mul_(sqrt(2 * (temperature) * lr / num_data)))

                alpha = lr if maximize else -lr
                param.add_(d_p, alpha=alpha)

        return loss
