import torch
from torch.optim import Optimizer
from math import sqrt


class SLGD(Optimizer):

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, temperature: float = 1):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, temperature=temperature)
        super(SLGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
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
            momentum_buffer_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            temperature = group['temperature']

            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = momentum_buffer_list[i]

                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                    # add tempered noise
                    if temperature > 0:
                        d_p = d_p.add(torch.randn_like(d_p).mul_(sqrt(2 * (1 - momentum) * temperature)))

                alpha = lr if maximize else -lr
                param.add_(d_p, alpha=alpha)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
