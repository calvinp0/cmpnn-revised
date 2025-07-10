import torch
import torch.nn as nn
class BaseMetric(nn.Module):
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
