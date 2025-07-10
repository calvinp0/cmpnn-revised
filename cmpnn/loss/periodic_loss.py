import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Base class for periodic losses handling sin/cos representations
class PeriodicLoss(nn.Module):
    """
    Base class to compute periodic losses in sin/cos form.
    """
    def __init__(self,
                 ignore_value: float = -10,
                 init_gamma: float = 1.0,
                 epsilon: float = 1e-6):
        super().__init__()
        self.ignore_value = ignore_value
        self.epsilon = epsilon
        # Log-space unconstrained gamma parameter; softplus enforces gamma > 0
        init_val = torch.tensor(init_gamma + epsilon)
        self._gamma_unconstrained = nn.Parameter(torch.log(init_val))

    def _unit_circle_penalty(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        penalty_terms = []
        pred_sin, pred_cos = self._sin_cos(preds)
        target_sin, target_cos = self._sin_cos(targets)
        valid = (target_sin != self.ignore_value) & (target_cos != self.ignore_value)
        if not valid.any():
            return torch.tensor(0.0, device=preds.device)
        # Compute penalties for sin and cos components
        mag = pred_sin[valid]**2 + pred_cos[valid]**2
        penalty_terms.append(((mag - 1) ** 2).mean())
        if len(penalty_terms) == 0:
            return torch.tensor(0.0, device=preds.device)
        gamma = F.softplus(self._gamma_unconstrained)
        
        return gamma * torch.stack(penalty_terms).mean()

    
    def _sin_cos(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts sine and cosine components from the last dimension of a tensor.

        Args:
            x (torch.Tensor): Tensor of shape (batch, 2) containing [sin, cos].
        Returns:
            sin_x, cos_x: Separate 1D tensors for sine and cosine.
        """
        # Assume x has two columns: sin and cos
        sin_x = x[:, 0]
        cos_x = x[:, 1]
        return sin_x, cos_x

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for loss computation; implemented in subclasses.
        """
        raise NotImplementedError
    
    def evaluation(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for evaluation metric; implemented in subclasses.
        """
        raise NotImplementedError


# MSE on chord (straight-line) distance between predicted and true unit-vectors
class PeriodicLossMSE(PeriodicLoss):
    """
    Computes Mean Squared Error for periodic targets via chord-length.
    """
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Split into sin/cos components
        sin_pred, cos_pred = self._sin_cos(preds)
        sin_true, cos_true = self._sin_cos(targets)

        # Create mask for valid entries
        mask = ((sin_true != self.ignore_value) &
                (cos_true != self.ignore_value))

        # Compute cosine of angular difference
        cos_delta = (sin_pred * sin_true + cos_pred * cos_true)

        # Loss = (1 - cos(delta))
        loss = (1.0 - cos_delta)[mask].mean()

        # add learned unit-circle regularization
        unit_penalty = self._unit_circle_penalty(preds, targets) if self.training else 0.0

        return loss + unit_penalty
    
    def evaluation(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sin_pred, cos_pred = self._sin_cos(preds)
        sin_true, cos_true = self._sin_cos(targets)
        # Filter out invalid entries
        valid = (sin_true != self.ignore_value) & (cos_true != self.ignore_value)

        # Recover wrapped angles in radians: range (-π, π]
        theta_pred = torch.atan2(sin_pred[valid], cos_pred[valid])
        theta_true = torch.atan2(sin_true[valid], cos_true[valid])
        delta = (theta_pred - theta_true + math.pi) % (2 * math.pi) - math.pi

        # Convert to degrees squared for interpretability
        delta_deg = delta * 180.0 / math.pi
        return torch.mean(delta_deg ** 2)

class PeriodicLossMAE(PeriodicLoss):
    def __init__(self,
                ignore_value: float = -10,
                init_gamma: float = 1.0,
                epsilon: float = 1e-6):
        super().__init__(ignore_value, init_gamma, epsilon)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sin_pred, cos_pred = preds[:, 0], preds[:, 1]
        sin_true, cos_true = targets[:, 0], targets[:, 1]
        valid = (sin_true != self.ignore_value) & (cos_true != self.ignore_value)
        if not valid.any():
            return torch.tensor(0.0, device=preds.device)
        theta_pred = torch.atan2(sin_pred[valid], cos_pred[valid])
        theta_true = torch.atan2(sin_true[valid], cos_true[valid])
        delta = theta_pred - theta_true
        delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi
        loss = (1 - torch.cos(delta)).pow(2).mean()
        if self.training:
            unit_penalty = self._unit_circle_penalty(preds, targets)
            loss = loss + unit_penalty
        return loss.mean()
    
    def evaluation(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sin_pred, cos_pred = preds[:, 0], preds[:, 1]
        sin_true, cos_true = targets[:, 0], targets[:, 1]
        valid = (sin_true != self.ignore_value) & (cos_true != self.ignore_value)
        if not valid.any():
            return torch.tensor(0.0, device=preds.device)

        theta_pred = torch.atan2(sin_pred[valid], cos_pred[valid])
        theta_true = torch.atan2(sin_true[valid], cos_true[valid])
        
        delta = (theta_pred - theta_true + math.pi) % (2 * math.pi) - math.pi
        delta_deg = delta * 180.0 / math.pi

        mae = delta_deg.abs().mean()
        return mae
    

# RMSE by taking sqrt of MSE
class PeriodicLossRMSE(PeriodicLossMSE):
    """
    Computes Root Mean Squared Error based on PeriodicLossMSE.
    """
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Use parent MSE and apply square root
        return torch.sqrt(super().forward(preds, targets))

    def evaluation(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # RMSE in evaluation = sqrt of MSE in degrees²
        return torch.sqrt(super().evaluation(preds, targets))


# # MAE on chord distance for training; true angular MAE in degrees for eval
# class PeriodicLossMAE(PeriodicLoss):
#     """
#     Computes Mean Absolute Error for periodic targets.
#     """
#     def __init__(self,
#                  ignore_value: float = -10,
#                  init_gamma: float = 1.0,
#                  epsilon: float = 1e-6):
#         super().__init__(ignore_value, init_gamma, epsilon)

#     def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         sin_pred, cos_pred = self._sin_cos(preds)
#         sin_true, cos_true = self._sin_cos(targets)
#         # Validity mask
#         mask = ((sin_true != self.ignore_value) & (cos_true != self.ignore_value)).float()

#         # Chord-length = sqrt((Δsin)^2 + (Δcos)^2)
#         diff_sq = (sin_pred - sin_true)**2 + (cos_pred - cos_true)**2
#         chord = torch.sqrt(diff_sq + self.epsilon)
#         mae = (chord * mask).sum() / mask.sum().clamp(min=1.0)
#         unit_penalty = self._unit_circle_penalty(preds, targets) if self.training else 0.0
#         return mae + unit_penalty

#     def evaluation(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         sin_pred, cos_pred = self._sin_cos(preds)
#         sin_true, cos_true = self._sin_cos(targets)
#         valid = (sin_true != self.ignore_value) & (cos_true != self.ignore_value)
#         if not valid.any():
#             return torch.tensor(0.0, device=preds.device)

#         # Recover wrapped angle differences
#         theta_pred = torch.atan2(sin_pred[valid], cos_pred[valid])
#         theta_true = torch.atan2(sin_true[valid], cos_true[valid])
#         delta = (theta_pred - theta_true + math.pi) % (2 * math.pi) - math.pi

#         # Return MAE in degrees for readability
#         return torch.mean(torch.abs(delta)) * 180.0 / math.pi


# R² metric based on true arc-length squared
class PeriodicLossR2(PeriodicLoss):
    """
    Computes coefficient of determination (R²) for periodic targets.
    """

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sin_pred, cos_pred = self._sin_cos(preds)
        sin_true, cos_true = self._sin_cos(targets)

        valid = (sin_true != self.ignore_value) & (cos_true != self.ignore_value)
        if not valid.any():
            # No valid data → perfect (edge case)
            return torch.tensor(0.0, device=preds.device)

        # Recover and wrap angle difference for SSR
        theta_pred = torch.atan2(sin_pred[valid], cos_pred[valid])
        theta_true = torch.atan2(sin_true[valid], cos_true[valid])
        delta_pred = (theta_pred - theta_true + math.pi) % (2 * math.pi) - math.pi
        ss_res = torch.sum(delta_pred ** 2)

        # Compute circular mean of true angles for SST
        mean_s = torch.mean(sin_true[valid])
        mean_c = torch.mean(cos_true[valid])
        theta_mean = torch.atan2(mean_s, mean_c)
        delta_true = (theta_true - theta_mean + math.pi) % (2 * math.pi) - math.pi
        ss_tot = torch.sum(delta_true ** 2).clamp(min=1e-6)

        # Return 1 - SSR/SST (note: higher is better)
        print(f"[DEBUG] SSR: {ss_res.item():.4f}, SST: {ss_tot.item():.4f}")
        return 1.0 - ss_res / ss_tot

    def evaluation(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # For loggable metric, same as forward
        return self.forward(preds, targets)

class PeriodicLossVonMises(PeriodicLoss):
    def __init__(self, ignore_value: float = -10.0, init_kappa: float = 1, kappa_target: float = 50.0, warmup_epochs: int = 30, init_gamma: float = 1.0, epsilon: float = 1e-6):
        super().__init__(ignore_value=ignore_value, init_gamma=init_gamma, epsilon=epsilon)
        self.kappa_target = kappa_target
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0  # Will be updated manually

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, reduction: str= None) -> torch.Tensor:
        sin_pred, cos_pred = self._sin_cos(preds)
        sin_true, cos_true = self._sin_cos(targets)
        mask = (sin_true != self.ignore_value) & (cos_true != self.ignore_value)
        if not mask.any():
            return torch.tensor(0.0, device=preds.device)

        theta_pred = torch.atan2(sin_pred[mask], cos_pred[mask])
        theta_true = torch.atan2(sin_true[mask], cos_true[mask])
        delta = (theta_pred - theta_true + torch.pi) % (2 * torch.pi) - torch.pi

        # Warmup κ based on current epoch
        factor = min(1.0, self.current_epoch / self.warmup_epochs)
        kappa = torch.tensor(1.0 + factor * (self.kappa_target - 1.0), device=preds.device)

        logI0 = torch.special.i0e(kappa).log() + kappa
        const = torch.tensor(2.0 * math.pi, device=preds.device, dtype=preds.dtype)
        log_norm = torch.log(const) + logI0

        nll = -kappa * torch.cos(delta) + log_norm


        penalty = self._unit_circle_penalty(preds, targets) if self.training else 0.0
        if reduction == 'none':
            return nll + penalty
        elif reduction == 'mean':
            return nll.mean() + penalty
        elif reduction == 'sum':
            return nll.sum() + penalty
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

import torch
import torch.nn as nn
import math

class AngularHuberLoss(nn.Module):
    def __init__(self, beta=1.0, ignore_value=-10.0, unit_weight=0.01):
        """
        Args:
            beta: Huber loss transition point (in degrees).
            ignore_value: Value to ignore in targets.
            unit_weight: Penalty for deviation from unit circle.
        """
        super().__init__()
        self.beta = beta
        self.ignore_value = ignore_value
        self.unit_weight = unit_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sin_pred, cos_pred = preds[:, 0], preds[:, 1]
        sin_true, cos_true = targets[:, 0], targets[:, 1]

        mask = (sin_true != self.ignore_value) & (cos_true != self.ignore_value)

        # Compute angular difference
        theta_pred = torch.atan2(sin_pred, cos_pred)
        theta_true = torch.atan2(sin_true, cos_true)
        delta = (theta_pred - theta_true + math.pi) % (2 * math.pi) - math.pi
        delta_deg = delta * 180.0 / math.pi

        error = delta_deg[mask].abs()

        # Huber loss
        quadratic = torch.clamp(error, max=self.beta)
        linear = (error - quadratic).clamp(min=0)
        loss = 0.5 * quadratic.pow(2) + self.beta * linear
        loss = loss.mean()

        # Unit circle penalty (soft)
        norm = (sin_pred ** 2 + cos_pred ** 2)
        unit_penalty = ((norm - 1) ** 2).mean()

        return loss + self.unit_weight * unit_penalty


class AngleMAELoss(nn.Module):
    def __init__(self, boost_threshold_deg=90, boost_weight=2.0):
        super().__init__()
        self.boost_threshold_rad = torch.deg2rad(torch.tensor(boost_threshold_deg))
        self.boost_weight = boost_weight

    def forward(self, pred_sincos, true_sincos):
        # pred_sincos and true_sincos are (N, 2) vectors
        pred_sin, pred_cos = pred_sincos[:, 0], pred_sincos[:, 1]
        true_sin, true_cos = true_sincos[:, 0], true_sincos[:, 1]

        mask = (true_sin != -10) & (true_cos != -10)
        if not mask.any():
            return torch.tensor(0.0, device=pred_sincos.device)
        
        pred_angles = torch.atan2(pred_sin[mask], pred_cos[mask])
        true_angles = torch.atan2(true_sin[mask], true_cos[mask])



        diff = torch.remainder(pred_angles - true_angles + torch.pi, 2 * torch.pi) - torch.pi
        abs_diff = torch.abs(diff)  # in radians

        boost_mask = (abs_diff > self.boost_threshold_rad).float()
        weights = 1.0 + (self.boost_weight - 1.0) * boost_mask

        angle_mae_deg = torch.rad2deg(abs_diff)
        weighted_mae = (angle_mae_deg * weights).mean()
        return weighted_mae
    

class HybridAngularLoss(PeriodicLoss):
    def __init__(self, tol_deg=3.0, ignore_value=-10.0, alpha=1.0, beta=1.0, adaptive_alpha=False, steepness=10.0):
        super().__init__(ignore_value=ignore_value)
        self.tol_deg = tol_deg
        self.beta = beta
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        self.steepness = steepness

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sin_pred, cos_pred = self._sin_cos(preds)
        sin_true, cos_true = self._sin_cos(targets)

        # Create mask for valid entries
        mask = (sin_true != self.ignore_value) & (cos_true != self.ignore_value)
        if not mask.any():
            return torch.tensor(0.0, device=preds.device)
        else:
            sin_pred = sin_pred[mask]
            cos_pred = cos_pred[mask]
            sin_true = sin_true[mask]
            cos_true = cos_true[mask]

        # Normalize predicted vectors
        sin_pred = sin_pred.clone()
        cos_pred = cos_pred.clone()

        pred_norm = torch.clamp(torch.sqrt(sin_pred**2 + cos_pred**2), min=1e-8)
        sin_pred = sin_pred / pred_norm
        cos_pred = cos_pred / pred_norm

        # Assumed sin_true and cos_true are already normalized
        cos_diff = cos_pred * cos_true + sin_pred * sin_true # cos(Δθ)
        cos_diff = torch.clamp(cos_diff, -1.0, 1.0)

        # Compute |(Δθ)| in degrees via atan2(sin Δθ, cos Δθ)
        sin_dff = sin_pred * cos_true - cos_pred * sin_true # sin(Δθ)
        delta_rad = torch.atan2(sin_dff, cos_diff) 
        delta_deg = torch.abs(delta_rad * (180.0 / torch.pi)) # |Δθ| in degrees

        # COsine loss: 1 - cos(Δθ)
        cos_loss = 1.0 - cos_diff # Already clipped

        # Angular MAE loss: |Δθ| in degrees
        if self.adaptive_alpha:
            # Adaptive alpha scaled by sigmoid centered at tol_deg
            weight = torch.sigmoid(self.steepness * (delta_deg - self.tol_deg))
        else:
            # Fixed alpha
            weight = self.alpha

        angular_mae = weight * delta_deg
        loss = self.beta * cos_loss + angular_mae
        return loss.mean()
    

class PureCosineLoss(PeriodicLoss):
    def __init__(self, ignore_value: float = -10.0, init_gamma: float = 1.0, epsilon: float = 1e-6):
        super().__init__(ignore_value=ignore_value, init_gamma=init_gamma, epsilon=epsilon)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sin_pred, cos_pred = self._sin_cos(preds)
        sin_true, cos_true = self._sin_cos(targets)

        valid = (sin_true != self.ignore_value) & (cos_true != self.ignore_value)
        if not valid.any():
            return torch.tensor(0.0, device=preds.device)

        # Normalize predicted vectors that are valid
        pred_norm = torch.clamp(torch.sqrt(sin_pred[valid]**2 + cos_pred[valid]**2), min=1e-8)
        sin_pred[valid] = sin_pred[valid] / pred_norm
        cos_pred[valid] = cos_pred[valid] / pred_norm

        # Compute cosine of angular difference
        cos_delta = (sin_pred * sin_true + cos_pred * cos_true)[valid]

        # Loss = 1 - cos(delta)
        loss = (1.0 - cos_delta).mean()

        # Add learned unit-circle regularization
        unit_penalty = self._unit_circle_penalty(preds, targets) if self.training else 0.0

        return loss + unit_penalty