import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMetric(nn.Module):
    """Abstract base for all custom loss/metric classes."""
    def __init__(self):
        super().__init__()


class MixedLossWrapper(BaseMetric):
    """Template for mixed-type losses/metrics: continuous, periodic (sin/cos), and angular."""
    def __init__(
        self,
        loss_type: str,
        cont_indices: list,
        per_indices: list,
        per_angle_indices: list,
        per_angle_180_indices: list = None,
        ignore_value: float = -10.0,
        gamma: float = 0.01,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.cont_indices = cont_indices
        self.per_indices = per_indices
        self.per_angle_indices = per_angle_indices
        self.per_angle_180_indices = per_angle_180_indices if per_angle_180_indices is not None else []
        self.ignore_value = ignore_value
        # unconstrained log-gamma for unit-circle penalty weight
        self._gamma_unconstrained = nn.Parameter(torch.log(torch.tensor(gamma)))

    def angular_distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # minimal absolute angle difference
        delta = (pred - target + 180) % 360 - 180
        return delta.abs()

    def angular_distance_180(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs()

    def unit_circle_penalty(self, preds: torch.Tensor) -> torch.Tensor:
        """Penalty = gamma * mean((sin^2+cos^2 - 1)^2) over valid sin/cos pairs."""
        if not self.per_indices:
            return torch.tensor(0.0, device=preds.device)
        penalty_terms = []
        for i in range(0, len(self.per_indices), 2):
            s, c = self.per_indices[i], self.per_indices[i+1]
            sin_v, cos_v = preds[:, s], preds[:, c]
            mask = (sin_v != self.ignore_value) & (cos_v != self.ignore_value)
            if mask.any():
                mag = sin_v[mask]**2 + cos_v[mask]**2
                penalty_terms.append(((mag - 1) ** 2).mean())
        if not penalty_terms:
            return torch.tensor(0.0, device=preds.device)
        gamma = F.softplus(self._gamma_unconstrained)
        return gamma * torch.stack(penalty_terms).mean()

    def compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        cont: bool = True,
        periodic: bool = True,
        angle_360: bool = True,
        angle_180: bool = False,
    ) -> torch.Tensor:
        """
        Aggregate loss/metric over all enabled target types.
        Subclasses must implement: _cont_loss, _angle_loss, _angle_scalar_loss.
        """
        losses = []

        if cont:
            for idx in self.cont_indices:
                mask = targets[:, idx] != self.ignore_value
                if not mask.any():
                    continue
                losses.append(self._cont_loss(preds[mask, idx], targets[mask, idx], idx))

        if periodic:
            for i in range(0, len(self.per_indices), 2):
                s, c = self.per_indices[i], self.per_indices[i+1]
                mask = (targets[:, s] != self.ignore_value) & (targets[:, c] != self.ignore_value)
                if not mask.any():
                    continue
                pred_th = torch.atan2(preds[mask, s], preds[mask, c])
                true_th = torch.atan2(targets[mask, s], targets[mask, c])
                losses.append(self._angle_loss(pred_th, true_th))

        if angle_360:
            for idx in self.per_angle_indices:
                mask = targets[:, idx] != self.ignore_value
                if not mask.any():
                    continue
                delta = self.angular_distance(preds[mask, idx], targets[mask, idx])
                losses.append(self._angle_scalar_loss(delta))
        if angle_180:
            for idx in self.per_angle_180_indices:
                mask = targets[:, idx] != self.ignore_value
                if not mask.any():
                    continue
                delta = self.angular_distance_180(preds[mask, idx], targets[mask, idx])
                losses.append(self._angle_scalar_loss_180(delta))


        if not losses:
            return torch.tensor(float('nan'), device=preds.device)
        return torch.stack(losses).mean()

    def compute_target_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        target_idx: int
    ) -> torch.Tensor:
        """Compute loss/metric for a single target index."""
        cont = target_idx in self.cont_indices
        periodic = target_idx in self.per_indices
        angle = target_idx in self.per_angle_indices
        return self.compute_loss(
            preds, targets,
            cont=cont,
            periodic=periodic,
            angle=angle,
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.compute_loss(preds, targets, cont=True, periodic=True, angle_360=False, angle_180=True)
        if self.training:
            # warn if evaluation mix-up
            print("[WARN] Using forward() in train/eval; penalty may be included")
        # include unit-circle only if any sin/cos present
        if any(
            ((targets[:, self.per_indices[i]] != self.ignore_value) &
             (targets[:, self.per_indices[i+1]] != self.ignore_value)).any()
            for i in range(0, len(self.per_indices), 2)
        ):
            return base + self.unit_circle_penalty(preds)
        return base

    def forward_test(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.compute_loss(preds, targets, cont=True, periodic=False, angle_360=True, angle_180=True)

    def forward_test_target(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        target_idx: int
    ) -> torch.Tensor:
        return self.compute_target_loss(preds, targets, target_idx)

    # Subclasses must implement these three:
    def _cont_loss(self, pred: torch.Tensor, target: torch.Tensor, idx: int) -> torch.Tensor:
        raise NotImplementedError

    def _angle_loss(
        self,
        pred_theta: torch.Tensor,
        true_theta: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def _angle_scalar_loss(self, delta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# === Concrete Loss/Metric Classes ===

class MixedMAELoss(MixedLossWrapper):
    def _cont_loss(self, pred, target, idx):
        # if idx == 2:
        #     return Target2LossWrapper(target_index=idx, ignore_value=self.ignore_value)(pred.unsqueeze(1), target.unsqueeze(1))
        return F.l1_loss(pred, target)

    def _angle_loss(self, pred_theta, true_theta):
        diff = 1 - torch.cos(pred_theta - true_theta)
        return torch.log1p(diff ** 2).mean()

    def _angle_scalar_loss(self, delta):
        rad = torch.deg2rad(delta)
        return torch.log1p((1 - torch.cos(rad)) ** 2).mean()


class MixedMSELoss(MixedMAELoss):
    def _cont_loss(self, pred, target, idx):
        # if idx == 2:
        #     return Target2LossWrapper(target_index=idx, ignore_value=self.ignore_value)(pred.unsqueeze(1), target.unsqueeze(1))
        return F.mse_loss(pred, target)


class MixedRMSELoss(MixedMSELoss):
    def forward(self, preds, targets):
        return torch.sqrt(super().forward(preds, targets))

    def forward_test(self, preds, targets):
        return torch.sqrt(super().forward_test(preds, targets))

    def forward_test_target(self, preds, targets, target_idx):
        return torch.sqrt(super().forward_test_target(preds, targets, target_idx))


class MixedR2Score(MixedLossWrapper):
    def _cont_loss(self, pred, target, idx):
        # if idx == 2:
        #     return Target2R2Score(target_index=idx, ignore_value=self.ignore_value)(pred.unsqueeze(1), target.unsqueeze(1))
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0, device=pred.device)

    def _angle_loss(self, pred_theta, true_theta):
        diff = 1 - torch.cos(pred_theta - true_theta)
        ss_res = (diff ** 2).sum()
        centered = true_theta - true_theta.mean()
        ss_tot = ((1 - torch.cos(centered)) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0, device=pred_theta.device)

    def _angle_scalar_loss(self, delta):
        rad = torch.deg2rad(delta)
        ss_res = ((1 - torch.cos(rad)) ** 2).sum()
        centered = torch.deg2rad(self.angular_distance(delta, delta.mean()))
        ss_tot = ((1 - torch.cos(centered)) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0, device=delta.device)


class MixedExplainedVarianceLoss(MixedR2Score):
    """Same pattern as R2 but uses explained variance formula."""
    def _cont_loss(self, pred, target, idx):
        # if idx =
        #     return Target2ExplainedVariance(target_index=idx, ignore_value=self.ignore_value)(pred.unsqueeze(1), target.unsqueeze(1))
        var_res = torch.var(target - pred)
        var_y = torch.var(target)
        return 1 - var_res / var_y if var_y > 0 else torch.tensor(0.0, device=pred.device)


class HybridLossWrapper(MixedLossWrapper):
    """Special two-part loss: log-gaussian NLL + KL reg for continuous, plus angular."""
    def __init__(
        self,
        loss_type: str = "hybrid",
        cont_indices: list = None,
        per_indices: list = None,
        per_angle_indices: list = None,
        ignore_value: float = -10.0,
        entropy_weight: float = 0.05,
        kl_weight: float = 0.01,
        kl_prior_std: float = 1.0,
    ):
        super().__init__(loss_type, cont_indices, per_indices, per_angle_indices, ignore_value)
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
        self.kl_prior_std = kl_prior_std

    def _cont_loss(self, pred, target, idx):
        # pred has shape (batch, 2): [mu, logvar]
        mu, logvar = pred.chunk(2, dim=-1)
        var = logvar.exp()
        # negative log‑likelihood term
        nll = 0.5 * (logvar + (target - mu).pow(2) / var).mean()

        # KL divergence term
        prior_var = self.kl_prior_std ** 2
        # log(prior variance)
        log_prior_var = torch.log(torch.tensor(prior_var, device=logvar.device))
        kl = 0.5 * (
            var     / prior_var        # σ²/σ₀²
        + mu.pow(2) / prior_var      # μ²/σ₀²
        - 1                          # constant
        - (logvar - log_prior_var)   # –log(σ²/σ₀²)
        ).mean()

        return nll + self.kl_weight * kl

    def _angle_loss(self, pred_theta, true_theta):
        diff = 1 - torch.cos(pred_theta - true_theta)
        return torch.log1p(diff ** 2).mean()

    def _angle_scalar_loss(self, delta):
        rad = torch.deg2rad(delta)
        return torch.log1p((1 - torch.cos(rad)) ** 2).mean()

    def _angle_scalar_loss_180(self, delta_norm: torch.Tensor) -> torch.Tensor:
        """
        Loss for a single bond-angle target confined to [0°, 180°].

        Parameters
        ----------
        delta_norm : Tensor
            (pred_norm − true_norm) in *normalised* units.

        Returns
        -------
        Tensor
            Mean Huber loss with ε = 5°,
            rescaled so gradient magnitude matches normalised targets.
        """
        # 1) de-normalise to degrees
        delta_deg = delta_norm * self.angle_std           # undo z-score
        # 2) absolute error (no wrap-around needed)
        abs_d = delta_deg.abs()
        # 3) Huber band ε = 5°
        eps  = 5.0
        quad = 0.5 * (abs_d.clamp(max=eps) / eps) ** 2    # quadratic inside band
        lin  = abs_d - eps / 2                            # linear outside
        loss = torch.where(abs_d <= eps, quad, lin)
        # 4) back-scale so gradients stay in the same ballpark
        return (loss / self.angle_std).mean()


# Registry and Factory
LOSS_CLASS_MAP = {
    'mixedmae': MixedMAELoss,
    'mixedmse': MixedMSELoss,
    'mixedrmse': MixedRMSELoss,
    'mixedr2': MixedR2Score,
    'mixedev': MixedExplainedVarianceLoss,
    'hybrid': HybridLossWrapper,
}


def MixedLossFactory(loss_type: str, cont_indices, per_indices, per_angle_indices, ignore_value=-10):
    cls = LOSS_CLASS_MAP.get(loss_type.lower())
    if cls is None:
        raise ValueError(f"Unsupported loss type '{loss_type}'.")
    return cls(
        loss_type=loss_type,
        cont_indices=cont_indices,
        per_indices=per_indices,
        per_angle_indices=per_angle_indices,
        ignore_value=ignore_value,
    )


# Target2 wrappers
class Target2LossWrapper(nn.Module):
    def __init__(self, target_index: int = 2, ignore_value: float = -10.0,
                 beta: float = 1.0, entropy_weight: float = 0.05):
        super().__init__()
        self.target_index = target_index
        self.ignore_value = ignore_value
        self.beta = beta
        self.entropy_weight = entropy_weight

    def forward(self, preds, targets):
        mask = targets[:, self.target_index] != self.ignore_value
        if not mask.any():
            return torch.tensor(float('nan'), device=preds.device)
        pred = preds[mask, self.target_index]
        true = targets[mask, self.target_index]
        huber = F.smooth_l1_loss(pred, true, beta=self.beta)
        entropy = -self.entropy_weight * torch.var(pred)
        return huber + entropy


class Target2R2Score(nn.Module):
    def __init__(self, target_index: int = 2, ignore_value: float = -10.0):
        super().__init__()
        self.target_index = target_index
        self.ignore_value = ignore_value

    def forward(self, preds, targets):
        mask = targets[:, self.target_index] != self.ignore_value
        if not mask.any():
            return torch.tensor(float('nan'), device=preds.device)
        y, y_pred = targets[mask, self.target_index], preds[mask, self.target_index]
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0, device=preds.device)


class Target2ExplainedVariance(nn.Module):
    def __init__(self, target_index: int = 2, ignore_value: float = -10.0):
        super().__init__()
        self.target_index = target_index
        self.ignore_value = ignore_value

    def forward(self, preds, targets):
        mask = targets[:, self.target_index] != self.ignore_value
        if not mask.any():
            return torch.tensor(float('nan'), device=preds.device)
        y, y_pred = targets[mask, self.target_index], preds[mask, self.target_index]
        var_y = torch.var(y)
        var_res = torch.var(y - y_pred)
        return 1 - var_res / var_y if var_y > 0 else torch.tensor(0.0, device=preds.device)


class MixedLossModule(nn.Module):
    """
    A custom loss module that combines different types of losses for continuous, periodic, and angular targets.
    This module is designed to be flexible and can be used in various machine learning tasks.
    """
    def __init__(self, loss_type: str, cont_indices: list, periodic_indices: list, angle_indices: list,
                 ignore_value: float = -10.0, gamma: float = 0.01):
        super().__init__()
        self.loss_type = loss_type
        self.cont_indices = cont_indices
        self.periodic_indices = periodic_indices
        self.angle_indices = angle_indices
        self.ignore_value = ignore_value
        self.gamma = gamma

    # 1. compute loss method
    # 2. periodic penalty method
    # 3. periodic angular distance method
    # 3. angular distance method (for 180°)
    