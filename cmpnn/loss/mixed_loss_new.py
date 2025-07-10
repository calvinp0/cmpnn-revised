import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple, Optional
from cmpnn.data.target import TargetSpec

class MixedMAELoss(nn.Module):
    """
    A MAE-Styled mixed loss that ierates over TargetSpec objects.
    """

    def __init__(self,
                 specs: List[TargetSpec],
                 ignore_value: float = -10.0,
                 ):
        super().__init__()
        self.specs = specs
        self.ignore_value = ignore_value

    @staticmethod
    def _periodic_scalar_mae(delta_deg: torch.Tensor):
        rad = torch.deg2rad(delta_deg)
        return torch.log1p((1 - torch.cos(rad)) ** 2).mean()
    
    @staticmethod
    def _angle_pair_mae(pred_theta: torch.Tensor, target_theta: torch.Tensor):
        cos_diff = 1 - torch.cos(pred_theta - target_theta)
        return torch.log1p(cos_diff ** 2).mean()
    
    @staticmethod
    def _wrap_deg(delta):
        return ((delta + 180) % 360) - 180
    
    def forward(self, preds, targets):
        """
        preds: (batch_size, num_targets)
        targets: (batch_size, num_targets)
        """
        loss, n = 0.0, 0

        for sp in self.specs:
            slc = sp.slc
            tgt = targets[:, slc]
            prd = preds[:, slc]

            if sp.kind == "cont":
                mask = tgt != self.ignore_value
                if mask.any():
                    total += F.l1_loss(prd[mask], tgt[mask])
                    n += 1

            elif sp.kind == "sincos":
                sin_t, cos_t = tgt[:, 0], tgt[:, 1]
                mask = (sin_t != self.ignore_value) & (cos_t != self.ignore_value)
                if mask.any():
                    θ_t = torch.atan2(sin_t[mask], cos_t[mask])
                    θ_p = torch.atan2(prd[mask, 0], prd[mask, 1])
                    total += self._angle_pair_mae(θ_p, θ_t)
                    n += 1

            elif sp.kind == "angle":
                mask = tgt != self.ignore_value
                if mask.any():
                    delta = self._wrap_deg(prd[mask] - tgt[mask]).abs()
                    total += self._periodic_scalar_mae(delta)
                    n += 1

        return total / n if n > 0 else torch.tensor(float("nan"), device=preds.device)
    
    def get_single(self,
                   key: Union[str, int],
                   preds: torch.Tensor,
                   targets: torch.Tensor) -> torch.Tensor:
        """
        key     : target *name* (str) **or** its ordinal index in self.specs
        returns : MAE‑style loss for that single target, NaN if no valid rows
        """
        # locate the spec
        if isinstance(key, str):
            idx = next(i for i, sp in enumerate(self.specs) if sp.name == key)
        else:
            idx = key
        sp  = self.specs[idx]
        slc = sp.slc
        tgt = targets[:, slc]
        prd = preds[:,  slc]

        if sp.kind in ("cont", "log"):
            mask = tgt != self.ignore_value
            return F.l1_loss(prd[mask], tgt[mask]) if mask.any() \
                   else torch.tensor(float("nan"), device=preds.device)

        elif sp.kind == "sincos":
            sin_t, cos_t = tgt[:, 0], tgt[:, 1]
            mask = (sin_t != self.ignore_value) & (cos_t != self.ignore_value)
            if not mask.any():
                return torch.tensor(float("nan"), device=preds.device)
            θ_t = torch.atan2(sin_t[mask], cos_t[mask])
            θ_p = torch.atan2(prd[mask, 0], prd[mask, 1])
            return self._angle_pair_mae(θ_p, θ_t)

        elif sp.kind == "angle":
            mask = tgt != self.ignore_value
            if not mask.any():
                return torch.tensor(float("nan"), device=preds.device)
            delta = self._wrap_deg(prd[mask] - tgt[mask]).abs()
            return self._periodic_scalar_mae(delta)
