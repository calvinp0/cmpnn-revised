import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousLoss(nn.Module):
    """
    Base class for continuous regression losses with optional ignore mask.

    Args:
        ignore_value (float): sentinel value in targets to skip.
    """
    def __init__(self, ignore_value: float = -10.0):
        super().__init__()
        self.ignore_value = ignore_value

    def _mask_valid(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask indicating positions where targets are valid (not equal to ignore_value).
        """
        return targets != self.ignore_value

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss. Implemented by subclasses.
        """
        raise NotImplementedError

    def evaluation(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute evaluation metric. Implemented by subclasses.
        """
        raise NotImplementedError


class ContinuousLossMAE(ContinuousLoss):
    """
    Mean Absolute Error (MAE) for continuous targets.
    Uses F.l1_loss under the hood, ignoring sentinel values.
    """
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = self._mask_valid(preds, targets)
        if not valid.any():
            # No valid points: return zero loss
            return torch.tensor(0.0, device=preds.device)
        # Compute MAE only on valid entries
        return F.l1_loss(preds[valid], targets[valid], reduction='mean')

    def evaluation(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # For MAE, training and evaluation are identical
        return self.forward(preds, targets)


class ContinuousLossMSE(ContinuousLoss):
    """
    Mean Squared Error (MSE) for continuous targets.
    Uses F.mse_loss, ignoring sentinel values.
    """
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = self._mask_valid(preds, targets)
        if not valid.any():
            return torch.tensor(0.0, device=preds.device)
        return F.mse_loss(preds[valid], targets[valid], reduction='mean')

    def evaluation(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # MSE metric same as training loss
        return self.forward(preds, targets)


class ContinuousLossRMSE(ContinuousLossMSE):
    """
    Root Mean Squared Error for continuous targets.
    RMSE = sqrt(MSE).
    """
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # sqrt of MSE on valid entries
        return torch.sqrt(super().forward(preds, targets))

    def evaluation(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # RMSE metric same as training
        return self.forward(preds, targets)


class ContinuousLossR2(ContinuousLoss):
    """
    Coefficient of determination (R²) for continuous regression.
    R² = 1 - SS_res / SS_tot, computed on valid entries.
    """
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = self._mask_valid(preds, targets)
        if not valid.any():
            return torch.tensor(0.0, device=preds.device)
        # residual sum of squares
        ss_res = torch.sum((targets[valid] - preds[valid]) ** 2)
        # total sum of squares around target mean
        mean_t = torch.mean(targets[valid])
        ss_tot = torch.sum((targets[valid] - mean_t) ** 2).clamp(min=1e-6)
        # R² can be negative if model is worse than mean predictor
        return 1.0 - ss_res / ss_tot

    def evaluation(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # R² metric same as training forward
        return self.forward(preds, targets)
