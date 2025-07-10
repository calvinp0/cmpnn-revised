import torch
import torch.nn as nn
import torchmetrics
from cmpnn.loss.base import BaseMetric

class MaskedRMSE(BaseMetric):
    def __init__(self, ignore_value=-10.0):
        super().__init__()
        self.metric = torchmetrics.MeanSquaredError(squared=False)
        self.ignore_value = ignore_value

    def forward(self, preds, targets):
        mask = targets != self.ignore_value
        if not mask.any():
            return torch.tensor(float("nan"), device=preds.device)
        return self.metric(preds[mask], targets[mask])

class MaskedMAE(BaseMetric):
    def __init__(self, ignore_value=-10.0):
        super().__init__()
        self.metric = torchmetrics.MeanAbsoluteError()
        self.ignore_value = ignore_value

    def forward(self, preds, targets):
        mask = targets != self.ignore_value
        if not mask.any():
            return torch.tensor(float("nan"), device=preds.device)
        return self.metric(preds[mask], targets[mask])


class MaskedR2Score(BaseMetric):
    def __init__(self, ignore_value=-10.0):
        super().__init__()
        self.metric = torchmetrics.R2Score()
        self.ignore_value = ignore_value

    def forward(self, preds, targets):
        mask = targets != self.ignore_value
        if not mask.any():
            return torch.tensor(float("nan"), device=preds.device)
        return self.metric(preds[mask], targets[mask])
    

class MaskedMSE(BaseMetric):
    def __init__(self, ignore_value=-10.0):
        super().__init__()
        self.metric = torchmetrics.MeanSquaredError()
        self.ignore_value = ignore_value

    def forward(self, preds, targets):
        mask = targets != self.ignore_value
        if not mask.any():
            return torch.tensor(float("nan"), device=preds.device)
        return self.metric(preds[mask], targets[mask])