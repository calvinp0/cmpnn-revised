from typing import List

import torch
import torch.nn as nn
from cmpnn.loss.continous_loss import ContinuousLossMAE, ContinuousLossMSE, ContinuousLossRMSE, ContinuousLossR2
from cmpnn.loss.periodic_loss import PeriodicLossMAE, PeriodicLossMSE, PeriodicLossRMSE, PeriodicLossR2

class CompositeLoss(nn.Module):
    """
    Composite loss/metric combining continuous and periodic targets,
    instantiated from metric names. Allows selecting training metric at forward call.

    Args:
        cont_indices (list[int]): Columns for continuous targets.
        per_indices (list[int]): Flat columns [s0,c0,s1,c1,...] for periodic targets.
        metrics (list[str]): List of metric names: one of ['MAE','MSE','RMSE','R2'].
        ignore_value (float): Sentinel in targets to skip.
    """
    def __init__(
        self,
        cont_indices: List[int],
        per_indices: List[int],
        metrics: List[str],
        ignore_value: float = -10.0
    ):
        super().__init__()
        allowed = {'MAE','MSE','RMSE','R2'}
        for m in metrics:
            if m not in allowed:
                raise ValueError(f"Unsupported metric '{m}'. Choose from {allowed}.")
        self.cont_indices = cont_indices
        self.per_indices = per_indices
        self.metrics = metrics
        self.ignore_value = ignore_value


        # Instantiate loss functions for each metric
        self.cont_fns = nn.ModuleDict()
        self.per_fns = nn.ModuleDict()
        CONT_METRIC_MAP = {
    'MAE': ContinuousLossMAE,
    'MSE': ContinuousLossMSE,
    'RMSE': ContinuousLossRMSE,
    'R2': ContinuousLossR2,
        }

        PER_METRIC_MAP = {
            'MAE': PeriodicLossMAE,
            'MSE': PeriodicLossMSE,
            'RMSE': PeriodicLossRMSE,
            'R2': PeriodicLossR2,
        }
        for m in metrics:
            if self.cont_indices:
                cont_cls = CONT_METRIC_MAP[m]
                self.cont_fns[m] = cont_cls(ignore_value=self.ignore_value)
            if self.per_indices:
                per_cls = PER_METRIC_MAP[m]
                self.per_fns[m] = per_cls(ignore_value=self.ignore_value, init_gamma=1.0, epsilon=1e-6)


    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        metric: str = None
    ) -> torch.Tensor:
        """
        Compute training loss for specified metric (default = first in metrics list).

        Args:
            preds   (Tensor): model outputs.
            targets (Tensor): ground truth.
            metric  (str): optional metric name to use for loss ('MAE','MSE',...).
        """
        key = metric.upper() if metric else self.metrics[0]
        if key not in self.metrics:
            raise ValueError(f"Metric '{key}' not in initialized metrics: {self.metrics}")

        total = torch.tensor(0.0, device=preds.device)
        if self.cont_indices:
            cp = preds[:, self.cont_indices]
            ct = targets[:, self.cont_indices]
            total = total + self.cont_fns[key](cp, ct)
        if self.per_indices:
            p_preds = preds[:, self.per_indices].view(preds.size(0), -1, 2)
            p_tgts  = targets[:, self.per_indices].view(targets.size(0), -1, 2)
            for i in range(p_preds.size(1)):
                total = total + self.per_fns[key](p_preds[:, i], p_tgts[:, i])
        return total

    def evaluation(self,
                   preds: torch.Tensor,
                   targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Compute evaluation metrics for all initialized metric names.

        Returns dict mapping each metric to its scalar value.
        """
        results = {}
        for m in self.metrics:
            total = torch.tensor(0.0, device=preds.device)
            if self.cont_indices:
                cp = preds[:, self.cont_indices]
                ct = targets[:, self.cont_indices]
                total = total + self.cont_fns[m].evaluation(cp, ct)
            if self.per_indices:
                p_preds = preds[:, self.per_indices].view(preds.size(0), -1, 2)
                p_tgts  = targets[:, self.per_indices].view(targets.size(0), -1, 2)
                for i in range(p_preds.size(1)):
                    total = total + self.per_fns[m].evaluation(p_preds[:, i], p_tgts[:, i])
            results[m] = total
        return results

    def __repr__(self):
        """
        Show selected metrics and the first loss functions for training.
        """
        repr_str = 'CompositeLoss(' + \
            f'cont_indices={self.cont_indices}, per_indices={self.per_indices}, ' + \
            'metrics=[' + ','.join(self.metrics) + ']'
        if self.cont_indices or self.per_indices:
            # show the training fns for the first metric
            first = self.metrics[0]
            repr_str += ', cont_fn=' + (
                self.cont_fns[first].__class__.__name__ if self.cont_indices else 'None'
            )
            repr_str += ', per_fn=' + (
                self.per_fns[first].__class__.__name__ if self.per_indices else 'None'
            )
        repr_str += ')'
        return repr_str
