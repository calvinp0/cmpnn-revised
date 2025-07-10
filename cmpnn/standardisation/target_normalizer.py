import pickle
import torch

class TargetNormalizer:
    def __init__(self, ignore_value: float = -10.0):
        self.means = None
        self.stds = None
        self.ignore_value = ignore_value

    def fit(self, ys: torch.Tensor):
        """
        Fit per-target mean and std, ignoring entries with ignore_value.
        ys: shape [n_samples, n_targets] or [n_samples]
        """
        if ys.ndim == 1:
            ys = ys.unsqueeze(1)

        mask = ys != self.ignore_value
        masked = ys * mask  # zero out ignored values

        count = mask.sum(dim=0)
        sum_ = masked.sum(dim=0)
        mean = sum_ / count

        # Variance: E[x^2] - (E[x])^2
        sum_sq = (masked ** 2).sum(dim=0)
        var = (sum_sq / count) - (mean ** 2)
        std = torch.sqrt(var)

        self.means = mean
        self.stds = std

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.means is None or self.stds is None:
            raise ValueError("TargetNormalizer not fitted.")
        device = y.device
        means = self.means.to(device)
        stds = self.stds.to(device)
        if y.ndim == 0:
            y = y.unsqueeze(0)
        return torch.where(y != self.ignore_value, (y - means) / stds, y)

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.means is None or self.stds is None:
            raise ValueError("TargetNormalizer not fitted.")
        device = y.device
        means = self.means.to(device)
        stds = self.stds.to(device)
        if y.ndim == 0:
            y = y.unsqueeze(0)
        return torch.where(y != self.ignore_value, y * stds + means, y)

    def save(self, path: str):
        state = {
            'means': self.means,
            'stds': self.stds,
            'ignore_value': self.ignore_value
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.means = state['means']
        self.stds = state['stds']
        self.ignore_value = state['ignore_value']
