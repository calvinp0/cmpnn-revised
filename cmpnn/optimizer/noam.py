import torch
from torch.optim.optimizer import Optimizer, required


class NoamLikeOptimizer(Optimizer):
    """Implements a Noam-like learning rate schedule with a piecewise linear warmup followed by exponential decay.
    
    This optimizer performs simple SGD updates using the scheduled learning rate.
    
    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        init_lr (float): Initial learning rate at step 0.
        max_lr (float): Maximum learning rate reached at the end of warmup.
        final_lr (float): Final learning rate to decay to at total_steps.
        warmup_steps (int): Number of steps for the linear warmup phase.
        total_steps (int): Total number of optimization steps.
    """

    def __init__(self, params, lr=required, init_lr=None, max_lr=1e-3, final_lr=1e-5, warmup_steps=1000,
                 total_steps=10000):
        if init_lr is None:
            init_lr = lr
        if init_lr < 0 or max_lr < 0 or final_lr < 0:
            raise ValueError("Learning rates must be non-negative")
        if warmup_steps < 1 or total_steps < warmup_steps:
            raise ValueError("warmup_steps must be at least 1 and total_steps must be >= warmup_steps")

        defaults = dict(init_lr=init_lr, max_lr=max_lr, final_lr=final_lr, warmup_steps=warmup_steps,
                        total_steps=total_steps)
        super(NoamLikeOptimizer, self).__init__(params, defaults)
        self._step_count = 0
        self._delta = (max_lr - init_lr) / warmup_steps
        self._lr_history = []

    def get_lr(self):
        i = self._step_count
        if i < self.defaults['warmup_steps']:
            lr = self.defaults['init_lr'] + self._delta * i
        else:
            gamma = (i - self.defaults['warmup_steps']) / (self.defaults['total_steps'] - self.defaults['warmup_steps'])
            lr = self.defaults['max_lr'] * (self.defaults['final_lr'] / self.defaults['max_lr']) ** gamma
        lr = max(self.defaults['final_lr'], lr)
        return lr

    def get_last_lr(self):
        return [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        current_lr = self.get_lr()
        self._step_count += 1
        self._lr_history.append(current_lr)

        with torch.no_grad():
            for group in self.param_groups:
                group['lr'] = current_lr
                for p in group['params']:
                    if p.grad is None or p.grad.is_sparse:
                        continue
                    p.add_(p.grad, alpha=-current_lr)

        return loss

    def plot_lr_schedule(self, save_path=None):
        import matplotlib.pyplot as plt
        plt.plot(self._lr_history)
        plt.title("Noam-like Learning Rate Schedule")
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
