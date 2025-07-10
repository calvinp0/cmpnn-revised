import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Base class for all custom loss metrics
class BaseMetric(nn.Module):
    def __init__(self):
        super().__init__()


# Wrapper to encapsulate metadata and dispatch logic for mixed-type losses
class MixedLossWrapper(BaseMetric):
    def __init__(self, loss_type: str, cont_indices, per_indices, per_angle_indices, ignore_value=-10, gamma: float = 0.01):
        super().__init__()
        self.loss_type = loss_type                      # Type of loss: e.g., 'mse', 'mae', etc.
        self.cont_indices = cont_indices                # Indices of continuous targets
        self.per_indices = per_indices                  # Indices of periodic targets (sin/cos pairs)
        self.per_angle_indices = per_angle_indices      # Indices of periodic scalar angles (degrees)
        self.ignore_value = ignore_value                # Value indicating missing/invalid targets
        #self.gamma = nn.Parameter(torch.tensor(gamma,
        #                                       dtype=torch.float32))# Penalty weight for sin^2 + cos^2 deviation
        self._gamma_unconstrained = nn.Parameter(torch.tensor(gamma).log().exp())  # Ensure it's positive

    def angular_distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = (pred - target + 180) % 360 - 180
        return delta.abs()

    def unit_circle_penalty(self, preds: torch.Tensor) -> torch.Tensor:
        if not self.per_indices:
            return torch.tensor(0.0, device=preds.device)
        penalty = 0.0
        count = 0
        for i in range(0, len(self.per_indices), 2):
            sin_idx, cos_idx = self.per_indices[i], self.per_indices[i + 1]
            sin_vals = preds[:, sin_idx]
            cos_vals = preds[:, cos_idx]
            mask = (sin_vals != self.ignore_value) & (cos_vals != self.ignore_value)
            if mask.any():
                magnitude = sin_vals[mask] ** 2 + cos_vals[mask] ** 2
                penalty += ((magnitude - 1) ** 2).mean()
                count += 1
        gamma = F.softplus(self._gamma_unconstrained)
        self.logged_gamma = gamma.item()
        return gamma * (penalty / count) if count > 0 else torch.tensor(0.0, device=preds.device)

    def compute_loss(self, preds, targets, cont=True, periodic=True, angle=True):
        raise NotImplementedError("Subclasses must implement compute_loss method.")

    def forward(self, preds, targets):
        base_loss = self.compute_loss(preds, targets, cont=True, periodic=True, angle=False)
        apply_penalty = any(
            ((targets[:, self.per_indices[i]] != self.ignore_value) &
            (targets[:, self.per_indices[i+1]] != self.ignore_value)).any()
            for i in range(0, len(self.per_indices), 2)
        )
        if not self.training:
                    print("[WARN] MixedLossWrapper.forward() used in evaluation mode. Consider using forward_test() to exclude unit circle penalty.")
        if apply_penalty:
            return base_loss + self.unit_circle_penalty(preds)
        return base_loss

    def forward_test(self, preds, targets):
        return self.compute_loss(preds, targets, cont=True, periodic=False, angle=True)

    def forward_test_target(self, preds, targets, target_idx):
        mask = None
        if target_idx in self.cont_indices:
            idx = self.cont_indices.index(target_idx)
            mask = targets[:, idx] != self.ignore_value
            return self._cont_loss(preds[:, idx][mask], targets[:, idx][mask])

        elif target_idx in self.per_angle_indices:
            idx = target_idx
            mask = targets[:, idx] != self.ignore_value
            delta = self.angular_distance(preds[:, idx][mask], targets[:, idx][mask])
            return self._angle_scalar_loss(delta)

        else:
            i = self.per_indices.index(target_idx)
            if i % 2 != 0:
                i -= 1  # make sure we grab the sin index
            sin_idx, cos_idx = self.per_indices[i], self.per_indices[i + 1]
            mask = (targets[:, sin_idx] != self.ignore_value) & (targets[:, cos_idx] != self.ignore_value)
            pred_theta = torch.atan2(preds[:, sin_idx][mask], preds[:, cos_idx][mask])
            true_theta = torch.atan2(targets[:, sin_idx][mask], targets[:, cos_idx][mask])
            return self._angle_loss(pred_theta, true_theta)

    def _cont_loss(self, pred, target, idx=None):
        raise NotImplementedError

    def _angle_loss(self, pred_theta, true_theta):
        raise NotImplementedError

    def _angle_scalar_loss(self, delta):
        raise NotImplementedError


# Loss: Mean Absolute Error
class MixedMAELoss(MixedLossWrapper):
    def _cont_loss(self, pred, target, idx=None):
        # For index == 2 (your t2), apply Huber or log-smooth
        if idx == 2:
            custom_loss = Target2LossWrapper(target_index=idx, ignore_value=self.ignore_value)
            return custom_loss(pred, target)
        else:
            return F.l1_loss(pred, target)

    def _angle_loss(self, pred_theta, true_theta):
        cos_diff = 1 - torch.cos(pred_theta - true_theta)
        return torch.log1p(cos_diff ** 2).mean()

    def _angle_scalar_loss(self, delta):
        rad = torch.deg2rad(delta)
        return torch.log1p((1 - torch.cos(rad)) ** 2).mean()

    def compute_loss(self, preds, targets, cont=True, periodic=True, angle=True):
        total_loss = 0.0
        count = 0

        if cont:
            for idx in self.cont_indices:
                mask = targets[:, idx] != self.ignore_value
                if mask.any():
                    if idx == 2:
                        total_loss += Target2LossWrapper(target_index=idx, ignore_value=self.ignore_value)(preds, targets)
                    else:
                        total_loss += self._cont_loss(preds[mask, idx], targets[mask, idx], idx)
                    count += 1

        if periodic:
            for i in range(0, len(self.per_indices), 2):
                sin_idx, cos_idx = self.per_indices[i], self.per_indices[i + 1]
                mask = (targets[:, sin_idx] != self.ignore_value) & (targets[:, cos_idx] != self.ignore_value)
                if mask.any():
                    pred_theta = torch.atan2(preds[mask, sin_idx], preds[mask, cos_idx])
                    true_theta = torch.atan2(targets[mask, sin_idx], targets[mask, cos_idx])
                    total_loss += self._angle_loss(pred_theta, true_theta)
                    count += 1

        if angle:
            for idx in self.per_angle_indices:
                mask = targets[:, idx] != self.ignore_value
                if mask.any():
                    delta = self.angular_distance(preds[mask, idx], targets[mask, idx])
                    total_loss += self._angle_scalar_loss(delta)
                    count += 1

        return total_loss / count if count > 0 else torch.tensor(float('nan'), device=preds.device)


# Loss: Mean Square Error
class MixedMSELoss(MixedMAELoss):
    def _cont_loss(self, pred, target, idx=None):
        if idx == 2:
            custom_loss = Target2LossWrapper(target_index=idx, ignore_value=self.ignore_value)
            return custom_loss(pred, target)
        return F.mse_loss(pred, target)

    def _angle_loss(self, pred_theta, true_theta):
        cos_diff = 1 - torch.cos(pred_theta - true_theta)
        return torch.log1p(cos_diff ** 2).mean()

    def _angle_scalar_loss(self, delta):
        rad = torch.deg2rad(delta)
        return torch.log1p((1 - torch.cos(rad)) ** 2).mean()


# Loss: Root Mean Square Error
class MixedRMSELoss(MixedMSELoss):
    def forward(self, preds, targets):
        return torch.sqrt(super().forward(preds, targets))

    def forward_test(self, preds, targets):
        return torch.sqrt(super().forward_test(preds, targets))
    
    def forward_test_target(self, preds, targets, target_idx):
        return torch.sqrt(super().forward_test_target(preds, targets, target_idx))


# Loss: R² Score
class MixedR2Score(MixedLossWrapper):

    def compute_loss(self, preds, targets, cont=True, periodic=True, angle=True):
        r2_sum = 0.0
        count = 0

        if cont:
            for idx in self.cont_indices:
                mask = targets[:, idx] != self.ignore_value
                if mask.any():
                    y = targets[mask, idx]
                    y_pred = preds[mask, idx]
                    if idx == 2:
                        print(f"Using custom R² for index {idx}")
                        r2_sum += Target2R2Score(target_index=idx, ignore_value=self.ignore_value)(preds, targets)
                    else:
                        ss_res = ((y - y_pred) ** 2).sum()
                        ss_tot = ((y - y.mean()) ** 2).sum()
                        r2_sum += 1 - ss_res / ss_tot
                    count += 1

        if periodic:
            for i in range(0, len(self.per_indices), 2):
                sin_idx, cos_idx = self.per_indices[i], self.per_indices[i + 1]
                mask = (targets[:, sin_idx] != self.ignore_value) & (targets[:, cos_idx] != self.ignore_value)
                if mask.any():
                    pred_theta = torch.atan2(preds[mask, sin_idx], preds[mask, cos_idx])
                    true_theta = torch.atan2(targets[mask, sin_idx], targets[mask, cos_idx])
                    error = 1 - torch.cos(pred_theta - true_theta)
                    ss_res = (error ** 2).sum()
                    centered = true_theta - true_theta.mean()
                    ss_tot = ((1 - torch.cos(centered)) ** 2).sum()
                    r2_sum += 1 - ss_res / ss_tot
                    count += 1

        if angle:
            for idx in self.per_angle_indices:
                mask = targets[:, idx] != self.ignore_value
                if mask.any():
                    y = targets[mask, idx]
                    y_pred = preds[mask, idx]
                    delta = self.angular_distance(y_pred, y)
                    rad = torch.deg2rad(delta)
                    centered = torch.deg2rad(self.angular_distance(y, y.mean()))
                    ss_res = ((1 - torch.cos(rad)) ** 2).sum()
                    ss_tot = ((1 - torch.cos(centered)) ** 2).sum()
                    r2_sum += 1 - ss_res / ss_tot
                    count += 1

        return r2_sum / count if count > 0 else torch.tensor(float('nan'), device=preds.device)
    
    def forward(self, preds, targets):
        return self.compute_loss(preds, targets, cont=True, periodic=True, angle=False)

    def forward_test(self, preds, targets):
        return self.compute_loss(preds, targets, cont=True, periodic=False, angle=True)

    def forward_test_target(self, preds, targets, target_idx):
        if target_idx in self.cont_indices:
            if target_idx == 2:
                return Target2R2Score(target_index=target_idx, ignore_value=self.ignore_value)(preds, targets)
            idx = self.cont_indices.index(target_idx)
            mask = targets[:, idx] != self.ignore_value
            y = targets[mask, idx]
            y_pred = preds[mask, idx]
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            return 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0, device=preds.device)

        elif target_idx in self.per_angle_indices:
            idx = target_idx
            mask = targets[:, idx] != self.ignore_value
            y = targets[mask, idx]
            y_pred = preds[mask, idx]
            rad = torch.deg2rad(self.angular_distance(y_pred, y))
            ss_res = ((1 - torch.cos(rad)) ** 2).sum()

            mean_rad = torch.deg2rad(self.angular_distance(y, y.mean()))
            ss_tot = ((1 - torch.cos(mean_rad)) ** 2).sum()
            return 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0, device=preds.device)

        else:
            # Assume target_idx is sin component in sin/cos pair
            i = self.per_indices.index(target_idx)
            if i % 2 != 0:
                i -= 1  # Align to sin index
            sin_idx, cos_idx = self.per_indices[i], self.per_indices[i + 1]
            mask = (targets[:, sin_idx] != self.ignore_value) & (targets[:, cos_idx] != self.ignore_value)
            pred_theta = torch.atan2(preds[:, sin_idx][mask], preds[:, cos_idx][mask])
            true_theta = torch.atan2(targets[:, sin_idx][mask], targets[:, cos_idx][mask])
            error = 1 - torch.cos(pred_theta - true_theta)
            ss_res = (error ** 2).sum()
            centered = true_theta - true_theta.mean()
            ss_tot = ((1 - torch.cos(centered)) ** 2).sum()
            return 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0, device=preds.device)
            

# Loss: Explained Variance
class MixedExplainedVarianceLoss(MixedLossWrapper):

    def compute_loss(self, preds, targets, cont=True, periodic=True, angle=True):
        var_sum = 0.0
        count = 0

        if cont:
            for idx in self.cont_indices:
                mask = targets[:, idx] != self.ignore_value
                if mask.any():
                    if idx == 2:
                        var_sum += Target2ExplainedVariance(target_index=idx, ignore_value=self.ignore_value)(preds, targets)
                    else:
                        y = targets[mask, idx]
                        y_pred = preds[mask, idx]
                        var_res = torch.var(y - y_pred)
                        var_y = torch.var(y)
                        var_sum += 1 - var_res / var_y
                    count += 1

        if periodic:
            for i in range(0, len(self.per_indices), 2):
                sin_idx, cos_idx = self.per_indices[i], self.per_indices[i + 1]
                mask = (targets[:, sin_idx] != self.ignore_value) & (targets[:, cos_idx] != self.ignore_value)
                if mask.any():
                    true_theta = torch.atan2(targets[mask, sin_idx], targets[mask, cos_idx])
                    pred_theta = torch.atan2(preds[mask, sin_idx], preds[mask, cos_idx])
                    diff = torch.atan2(torch.sin(pred_theta - true_theta), torch.cos(pred_theta - true_theta))
                    var_diff = torch.var(diff)
                    var_true = torch.var(true_theta)
                    var_sum += 1 - var_diff / var_true
                    count += 1

        if angle:
            for idx in self.per_angle_indices:
                mask = targets[:, idx] != self.ignore_value
                if mask.any():
                    delta = self.angular_distance(preds[mask, idx], targets[mask, idx])
                    rad = torch.deg2rad(delta)
                    var_diff = torch.var(1 - torch.cos(rad))
                    var_y = torch.var(1 - torch.cos(torch.deg2rad(targets[mask, idx])))
                    var_sum += 1 - var_diff / var_y
                    count += 1

        return var_sum / count if count > 0 else torch.tensor(float('nan'), device=preds.device)

    
    def forward(self, preds, targets):
        return self.compute_loss(preds, targets, cont=True, periodic=True, angle=False)
    
    def forward_test(self, preds, targets):
        return self.compute_loss(preds, targets, cont=True, periodic=False, angle=True)

    def forward_test_target(self, preds, targets, target_idx):
        if target_idx in self.cont_indices:
            if target_idx == 2:
                return Target2ExplainedVariance(target_index=2, ignore_value=self.ignore_value)(preds, targets)
            idx = target_idx
            mask = targets[:, idx] != self.ignore_value
            y = targets[mask, idx]
            y_pred = preds[mask, idx]
            var_y = torch.var(y)
            var_res = torch.var(y - y_pred)
            return 1 - var_res / var_y if var_y > 0 else torch.tensor(0.0, device=preds.device)

        elif target_idx in self.per_angle_indices:
            idx = target_idx
            mask = targets[:, idx] != self.ignore_value
            y = targets[mask, idx]
            y_pred = preds[mask, idx]
            delta = self.angular_distance(y_pred, y)
            rad = torch.deg2rad(delta)
            var_diff = torch.var(1 - torch.cos(rad))
            rad_true = torch.deg2rad(self.angular_distance(y, y.mean()))
            var_y = torch.var(1 - torch.cos(rad_true))
            return 1 - var_diff / var_y if var_y > 0 else torch.tensor(0.0, device=preds.device)

        else:
            i = self.per_indices.index(target_idx)
            if i % 2 != 0:
                i -=1
            sin_idx, cos_idx = self.per_indices[i], self.per_indices[i + 1]
            mask = (targets[:, sin_idx] != self.ignore_value) & (targets[:, cos_idx] != self.ignore_value)
            pred_theta = torch.atan2(preds[mask, sin_idx], preds[mask, cos_idx])
            true_theta = torch.atan2(targets[mask, sin_idx], targets[mask, cos_idx])
            diff = 1 - torch.cos(pred_theta - true_theta)
            var_diff = torch.var(diff)
            var_true = torch.var(1 - torch.cos(true_theta - true_theta.mean()))
            return 1 - var_diff / var_true if var_true > 0 else torch.tensor(0.0, device=preds.device)

class HybridLossWrapper(MixedLossWrapper):
    def __init__(
        self,
        loss_type: str = "hybrid",  # dummy to satisfy the factory
        cont_indices: list = None,
        per_indices: list = None,
        per_angle_indices: list = None,
        ignore_value: float = -10.0,
        entropy_weight: float = 0.05,
        kl_weight: float = 0.01,          # <== added this
        kl_prior_std: float = 1.0,        # <== added this too
    ):
        super().__init__(
            loss_type=loss_type,
            cont_indices=cont_indices,
            per_indices=per_indices,
            per_angle_indices=per_angle_indices,
            ignore_value=ignore_value,
        )
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
        self.kl_prior_std = kl_prior_std

    def log_gaussian_nll(self, mu, logvar, target):
        var = logvar.exp()
        return 0.5 * (logvar + (target - mu) ** 2 / var)
    
    def _angle_loss(self, pred_theta, true_theta):
        cos_diff = 1 - torch.cos(pred_theta - true_theta)
        return torch.log1p(cos_diff ** 2).mean()

    def _angle_scalar_loss(self, delta):
        rad = torch.deg2rad(delta)
        return torch.log1p((1 - torch.cos(rad)) ** 2).mean()

    def kl_regularization(self, logvar):
        prior_logvar = torch.log(torch.tensor(self.kl_prior_std ** 2, device=logvar.device))
        return 0.5 * ((logvar.exp() / self.kl_prior_std**2) + (prior_logvar - logvar) - 1).mean()



    def forward(self, preds, targets):
        total_loss = 0.0
        reg_loss = 0.0
        count = 0

        # Handle continuous targets with log-Gaussian NLL
        # Continuous targets: log-Gaussian NLL + regularize logvar
        # Log-Gaussian NLL + KL reg for continuous targets
        for i, idx in enumerate(self.cont_indices):
            mask = targets[:, i] != self.ignore_value
            if not mask.any():
                continue
            mu = preds[mask, 2 * i]
            logvar = preds[mask, 2 * i + 1]
            target = targets[mask, i]

            total_loss += self.log_gaussian_nll(mu, logvar, target).mean()
            reg_loss += self.kl_regularization(logvar)
            count += 1
        # Handle periodic angle targets using standard angular loss
        for i in range(0, len(self.per_indices), 2):
            sin_idx, cos_idx = self.per_indices[i], self.per_indices[i + 1]
            mask = (targets[:, sin_idx] != self.ignore_value) & (targets[:, cos_idx] != self.ignore_value)
            if not mask.any():
                continue
            pred_theta = torch.atan2(preds[mask, sin_idx], preds[mask, cos_idx])
            true_theta = torch.atan2(targets[mask, sin_idx], targets[mask, cos_idx])
            total_loss += self._angle_loss(pred_theta, true_theta)
            count += 1


        return total_loss / count if count > 0 else torch.tensor(float('nan'), device=preds.device)

    def forward_test(self, preds, targets):
        total_loss = 0.0
        count = 0

        # Each continuous target has one column: μ only
        for idx in self.cont_indices:
            mask = targets[:, idx] != self.ignore_value
            if not mask.any():
                continue
            pred_mu = preds[mask, idx]
            true_val = targets[mask, idx]
            total_loss += F.l1_loss(pred_mu, true_val)
            count += 1

        # Periodic targets are next in order
        for idx in self.per_angle_indices:
            mask = targets[:, idx] != self.ignore_value
            if mask.any():
                delta = self.angular_distance(preds[mask, idx], targets[mask, idx])
                total_loss += self._angle_scalar_loss(delta)
                count += 1

        return total_loss / count if count > 0 else torch.tensor(float('nan'), device=preds.device)

    def forward_test_target(self, preds, targets, target_idx):
        if target_idx in self.cont_indices:
            mask = targets[:, target_idx] != self.ignore_value
            if not mask.any():
                return torch.tensor(float('nan'), device=preds.device)
            pred_mu = preds[mask, target_idx]
            true_val = targets[mask, target_idx]
            return F.l1_loss(pred_mu, true_val)

        elif target_idx in self.per_angle_indices:
            idx = target_idx
            mask = targets[:, idx] != self.ignore_value
            delta = self.angular_distance(preds[:, idx][mask], targets[:, idx][mask])
            return self._angle_scalar_loss(delta)

        return torch.tensor(float('nan'), device=preds.device)
 
# Registry for all mixed loss types
LOSS_CLASS_MAP = {
    'mixedmae': MixedMAELoss,
    'mixedmse': MixedMSELoss,
    'mixedrmse': MixedRMSELoss,
    'mixedr2': MixedR2Score,
    'mixedexplainedvariance': MixedExplainedVarianceLoss,
    'hybrid': HybridLossWrapper,
}

# Factory function to instantiate the appropriate loss class

def MixedLossFactory(loss_type: str, cont_indices, per_indices, per_angle_indices, ignore_value=-10):
    loss_type = loss_type.lower()  # normalize input
    if loss_type not in LOSS_CLASS_MAP:
        raise ValueError(f"Unsupported loss type '{loss_type}'. Choose from {list(LOSS_CLASS_MAP.keys())}.")
    return LOSS_CLASS_MAP[loss_type](
        loss_type=loss_type,
        cont_indices=cont_indices,
        per_indices=per_indices,
        per_angle_indices=per_angle_indices,
        ignore_value=ignore_value,
    )


class Target2LossWrapper(nn.Module):
    def __init__(self, target_index: int = 2, ignore_value: float = -10.0,
                 beta: float = 1.0, entropy_weight: float = 0.05):
        super().__init__()
        self.target_index = target_index
        self.ignore_value = ignore_value
        self.beta = beta
        self.entropy_weight = entropy_weight

    def forward(self, preds, targets):
        idx = self.target_index
        mask = targets[:, idx] != self.ignore_value
        if not mask.any():
            return torch.tensor(float('nan'), device=preds.device)

        pred = preds[mask, idx]
        true = targets[mask, idx]

        # Huber Loss
        huber = F.smooth_l1_loss(pred, true, beta=self.beta)

        # Entropy boost: encourage some prediction spread
        entropy_penalty = -self.entropy_weight * torch.var(pred)

        return huber + entropy_penalty

class Target2R2Score(nn.Module):
    def __init__(self, target_index: int = 2, ignore_value: float = -10.0):
        super().__init__()
        self.target_index = target_index
        self.ignore_value = ignore_value

    def forward(self, preds, targets):
        idx = self.target_index
        mask = targets[:, idx] != self.ignore_value
        if not mask.any():
            return torch.tensor(float('nan'), device=preds.device)

        y = targets[mask, idx]
        y_pred = preds[mask, idx]

        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()

        return 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0, device=preds.device)


class Target2ExplainedVariance(nn.Module):
    def __init__(self, target_index: int = 2, ignore_value: float = -10.0):
        super().__init__()
        self.target_index = target_index
        self.ignore_value = ignore_value

    def forward(self, preds, targets):
        idx = self.target_index
        mask = targets[:, idx] != self.ignore_value
        if not mask.any():
            return torch.tensor(float('nan'), device=preds.device)

        y = targets[mask, idx]
        y_pred = preds[mask, idx]

        var_y = torch.var(y)
        var_res = torch.var(y - y_pred)

        return 1 - var_res / var_y if var_y > 0 else torch.tensor(0.0, device=preds.device)






# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from cmpnn.loss.base import BaseMetric


# class MixedLossWrapper(BaseMetric):
#     def __init__(self, loss_type: str, cont_indices, per_indices, per_angle_indices, ignore_value=-10):
#         super().__init__()
#         self.loss_type = loss_type
#         self.cont_indices = cont_indices
#         self.per_indices = per_indices
#         self.ignore_value = ignore_value
#         self.per_angle_indices = per_angle_indices

# class MixedMSELoss(MixedLossWrapper):
#     def __init__(self, cont_indices, per_indices, per_angle_indices, ignore_value=-10):
#         super().__init__('mse', cont_indices, per_indices, per_angle_indices, ignore_value)
#         self.loss = MixedMSELoss(cont_indices, per_indices, per_angle_indices, ignore_value)

#     def forward(self, preds, targets):
#         return self.loss(preds, targets)
#     def forward_test(self, preds, targets):
#         return self.loss.forward_test(preds, targets)
#     def forward_test_target(self, preds, targets, target_idx):
#         return self.loss.forward_test_target(preds, targets, target_idx)

# class MixedRMSELoss(MixedLossWrapper):
#     def __init__(self, cont_indices, per_indices, per_angle_indices, ignore_value=-10):
#         super().__init__('rmse', cont_indices, per_indices, per_angle_indices, ignore_value)
#         self.loss = MixedRMSELoss(cont_indices, per_indices, per_angle_indices, ignore_value)

#     def forward(self, preds, targets):
#         return self.loss(preds, targets)
#     def forward_test(self, preds, targets):
#         return self.loss.forward_test(preds, targets)
#     def forward_test_target(self, preds, targets, target_idx):
#         return self.loss.forward_test_target(preds, targets, target_idx)
    
# class MixedR2Score(MixedLossWrapper):
#     def __init__(self, cont_indices, per_indices, per_angle_indices, ignore_value=-10):
#         super().__init__('r2', cont_indices, per_indices, per_angle_indices, ignore_value)
#         self.loss = MixedR2Score(cont_indices, per_indices, per_angle_indices, ignore_value)

#     def forward(self, preds, targets):
#         return self.loss(preds, targets)
#     def forward_test(self, preds, targets):
#         return self.loss.forward_test(preds, targets)
#     def forward_test_target(self, preds, targets, target_idx):
#         return self.loss.forward_test_target(preds, targets, target_idx)
    
# class MixedMAELoss(MixedLossWrapper):
#     def __init__(self, cont_indices, per_indices, per_angle_indices, ignore_value=-10):
#         super().__init__('mae', cont_indices, per_indices, per_angle_indices, ignore_value)
#         self.loss = MixedMAELoss(cont_indices, per_indices, per_angle_indices, ignore_value)

#     def forward(self, preds, targets):
#         return self.loss(preds, targets)
#     def forward_test(self, preds, targets):
#         return self.loss.forward_test(preds, targets)
#     def forward_test_target(self, preds, targets, target_idx):
#         return self.loss.forward_test_target(preds, targets, target_idx)


# class MixedMSELoss(nn.Module):
#     def __init__(self, cont_indices, per_indices, per_angle_indices, ignore_value=-10.0):
#         super().__init__()
#         self.cont_indices = cont_indices
#         self.per_indices = per_indices
#         self.ignore_value = ignore_value
#         self.per_angle_indices = per_angle_indices

#     def forward(self, preds, targets):
#         if preds.ndim == 1:
#             preds = preds.unsqueeze(1)
#             targets = targets.unsqueeze(1)
#         total_loss = 0.0
#         count = 0

#         for idx in self.cont_indices:
#             mask = targets[:, idx] != self.ignore_value
#             if mask.any():
#                 loss = F.mse_loss(preds[mask, idx], targets[mask, idx])
#                 total_loss += loss
#                 count += 1

#         for i in range(0, len(self.per_indices), 2):
#             sin_idx = self.per_indices[i]
#             cos_idx = self.per_indices[i + 1]
#             mask = (targets[:, sin_idx] != self.ignore_value) & (targets[:, cos_idx] != self.ignore_value)
#             if mask.any():
#                 true_theta = torch.atan2(targets[mask, sin_idx], targets[mask, cos_idx])
#                 pred_theta = torch.atan2(preds[mask, sin_idx], preds[mask, cos_idx])
#                 angular_loss = 1 - torch.cos(pred_theta - true_theta)
#                 total_loss += angular_loss.mean()
#                 count += 1

#         if count == 0:
#             return torch.tensor(float('nan'), device=preds.device)
#         return total_loss / count

#     def forward_test(self, preds, targets):

#         # Used for test: assume periodic targets are scalar angles (1 per angle)
#         total_loss = 0.0
#         count = 0

#         # Continuous: MSE
#         for idx in self.cont_indices:
#             mask = targets[:, idx] != self.ignore_value
#             if mask.any():
#                 loss = F.mse_loss(preds[mask, idx], targets[mask, idx])
#                 total_loss += loss
#                 count += 1

#         # Periodic: angular loss
#         for idx in self.per_angle_indices:
#             mask = targets[:, idx] != self.ignore_value
#             if mask.any():
#                 # Convert to radians for angular difference
#                 pred_rad = torch.deg2rad(preds[mask, idx])
#                 true_rad = torch.deg2rad(targets[mask, idx])
#                 angular_loss = 1 - torch.cos(pred_rad - true_rad)
#                 total_loss += angular_loss.mean()
#                 count += 1

#         if count == 0:
#             return torch.tensor(float('nan'), device=preds.device)
#         return total_loss / count

#     def forward_test_target(self, preds, targets, target_idx):
#         if preds.ndim == 1:
#             preds = preds.unsqueeze(1)
#             targets = targets.unsqueeze(1)

#         if target_idx not in self.cont_indices and target_idx not in self.per_indices:
#             raise ValueError(f"Target index {target_idx} not in cont or periodic targets.")

#         mask = targets[:, target_idx] != self.ignore_value
#         if not mask.any():
#             return torch.tensor(float('nan'), device=preds.device)
        
#         if target_idx in self.cont_indices:
#             return F.mse_loss(preds[mask, target_idx], targets[mask, target_idx])
#         elif target_idx in self.per_angle_indices:
#             pred_rad = torch.deg2rad(preds[mask, target_idx])
#             true_rad = torch.deg2rad(targets[mask, target_idx])
#             return (1 - torch.cos(pred_rad - true_rad)).mean()


# class MixedRMSELoss(MixedMSELoss):
#     def forward(self, preds, targets):
#         mse = super().forward(preds, targets)
#         return torch.sqrt(mse)
#     def forward_test(self, preds, targets):
#         mse = super().forward_test(preds, targets)
#         return torch.sqrt(mse)
#     def forward_test_target(self, preds, targets, target_idx):
#         mse = super().forward_test_target(preds, targets, target_idx)
#         return torch.sqrt(mse)


# class MixedR2Score(nn.Module):
#     def __init__(self, cont_indices, per_indices, per_angle_indices, ignore_value=-10.0):
#         super().__init__()
#         self.cont_indices = cont_indices
#         self.per_indices = per_indices
#         self.ignore_value = ignore_value
#         self.per_angle_indices = per_angle_indices

#     def forward(self, preds, targets):
#         if preds.ndim == 1:
#             preds = preds.unsqueeze(1)
#             targets = targets.unsqueeze(1)
#         r2_sum = 0.0
#         count = 0

#         for idx in self.cont_indices:
#             mask = targets[:, idx] != self.ignore_value
#             if mask.any():
#                 target_vals = targets[mask, idx]
#                 pred_vals = preds[mask, idx]
#                 ss_res = ((target_vals - pred_vals) ** 2).sum()
#                 ss_tot = ((target_vals - target_vals.mean()) ** 2).sum()
#                 r2 = 1 - ss_res / ss_tot
#                 r2_sum += r2
#                 count += 1

#         for idx in (0, len(self.per_indices), 2):
#             sin_idx = self.per_indices[idx]
#             cos_idx = self.per_indices[idx + 1]
#             mask = (targets[:, sin_idx] != self.ignore_value) & (targets[:, cos_idx] != self.ignore_value)
#             if mask.any():
#                 true_theta = torch.atan2(targets[mask, sin_idx], targets[mask, cos_idx])
#                 pred_theta = torch.atan2(preds[mask, sin_idx], preds[mask, cos_idx])
#                 error = 1 - torch.cos(pred_theta - true_theta)
#                 ss_res = error.sum()
#                 ss_tot = (1 - torch.cos(true_theta - true_theta.mean())).sum()
#                 r2 = 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0, device=preds.device)
#                 r2_sum += r2
#                 count += 1


#         if count == 0:
#             return torch.tensor(float('nan'), device=preds.device)
#         return r2_sum / count

#     def forward_test(self, preds, targets):
#         r2_sum = 0.0
#         count = 0

#         # R² for continuous targets
#         for idx in self.cont_indices:
#             mask = targets[:, idx] != self.ignore_value
#             if mask.any():
#                 y_true = targets[mask, idx]
#                 y_pred = preds[mask, idx]
#                 ss_res = ((y_true - y_pred) ** 2).sum()
#                 ss_tot = ((y_true - y_true.mean()) ** 2).sum()
#                 r2 = 1 - ss_res / ss_tot
#                 r2_sum += r2
#                 count += 1

#         # Angular R² for periodic targets
#         for idx in self.per_angle_indices:
#             mask = targets[:, idx] != self.ignore_value
#             if mask.any():
#                 true_rad = torch.deg2rad(targets[mask, idx])
#                 pred_rad = torch.deg2rad(preds[mask, idx])
#                 error = 1 - torch.cos(pred_rad - true_rad)
#                 ss_res = error.sum()
#                 ss_tot = (1 - torch.cos(true_rad - true_rad.mean())).sum()
#                 r2 = 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0, device=preds.device)
#                 r2_sum += r2
#                 count += 1

#         if count == 0:
#             return torch.tensor(float('nan'), device=preds.device)
#         return r2_sum / count
    
#     def forward_test_target(self, preds, targets, target_idx):
#         if preds.ndim == 1:
#             preds = preds.unsqueeze(1)
#             targets = targets.unsqueeze(1)

#         if target_idx not in self.cont_indices and target_idx not in self.per_indices:
#             raise ValueError(f"Target index {target_idx} not in cont or periodic targets.")

#         mask = targets[:, target_idx] != self.ignore_value
#         if not mask.any():
#             return torch.tensor(float('nan'), device=preds.device)
        
#         if target_idx in self.cont_indices:
#             y_true = targets[mask, target_idx]
#             y_pred = preds[mask, target_idx]
#             ss_res = ((y_true - y_pred) ** 2).sum()
#             ss_tot = ((y_true - y_true.mean()) ** 2).sum()
#             return 1 - ss_res / ss_tot
#         elif target_idx in self.per_angle_indices:
#             true_rad = torch.deg2rad(targets[mask, target_idx])
#             pred_rad = torch.deg2rad(preds[mask, target_idx])
#             error = 1 - torch.cos(pred_rad - true_rad)
#             ss_res = error.sum()
#             ss_tot = (1 - torch.cos(true_rad - true_rad.mean())).sum()
#             return 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0, device=preds.device)
#         return torch.tensor(float('nan'), device=preds.device)


# class MixedMAELoss(nn.Module):
#     def __init__(self, cont_indices, per_indices, per_angle_indices, ignore_value=-10.0):
#         super().__init__()
#         self.cont_indices = cont_indices
#         self.per_indices = per_indices
#         self.ignore_value = ignore_value
#         self.per_angle_indices = per_angle_indices

#     def forward(self, preds, targets):
#         if preds.ndim == 1:
#             preds = preds.unsqueeze(1)
#             targets = targets.unsqueeze(1)
#         total_loss = 0.0
#         count = 0

#         # Continuous: MAE
#         for idx in self.cont_indices:
#             mask = targets[:, idx] != self.ignore_value
#             if mask.any():
#                 loss = F.l1_loss(preds[mask, idx], targets[mask, idx])
#                 total_loss += loss
#                 count += 1

#         # Periodic: Angular distance (same as MSE version)
#         for i in range(0, len(self.per_indices), 2):
#             sin_idx = self.per_indices[i]
#             cos_idx = self.per_indices[i + 1]
#             mask = (targets[:, sin_idx] != self.ignore_value) & (targets[:, cos_idx] != self.ignore_value)
#             if mask.any():
#                 true_theta = torch.atan2(targets[mask, sin_idx], targets[mask, cos_idx])
#                 pred_theta = torch.atan2(preds[mask, sin_idx], preds[mask, cos_idx])
#                 angular_loss = 1 - torch.cos(pred_theta - true_theta)
#                 total_loss += angular_loss.mean()
#                 count += 1

#         if count == 0:
#             return torch.tensor(float('nan'), device=preds.device)
#         return total_loss / count
    
#     def forward_test(self, preds, targets):
#         # Used for test: assume periodic targets are scalar angles (1 per angle)
#         total_loss = 0.0
#         count = 0

#         # Continuous: MAE
#         for idx in self.cont_indices:
#             mask = targets[:, idx] != self.ignore_value
#             if mask.any():
#                 loss = F.l1_loss(preds[mask, idx], targets[mask, idx])
#                 total_loss += loss
#                 count += 1

#         # Periodic: angular MAE using angle difference
#         for idx in self.per_angle_indices:
#             mask = targets[:, idx] != self.ignore_value
#             if mask.any():
#                 pred_rad = torch.deg2rad(preds[mask, idx])
#                 true_rad = torch.deg2rad(targets[mask, idx])
#                 diff = torch.atan2(torch.sin(pred_rad - true_rad), torch.cos(pred_rad - true_rad)).abs()
#                 total_loss += diff.mean()
#                 count += 1

#         if count == 0:
#             return torch.tensor(float('nan'), device=preds.device)
#         return total_loss / count

#     def forward_test_target(self, preds, targets, target_idx):
#         if preds.ndim == 1:
#             preds = preds.unsqueeze(1)
#             targets = targets.unsqueeze(1)

#         if target_idx not in self.cont_indices and target_idx not in self.per_indices:
#             raise ValueError(f"Target index {target_idx} not in cont or periodic targets.")

#         mask = targets[:, target_idx] != self.ignore_value
#         if not mask.any():
#             return torch.tensor(float('nan'), device=preds.device)
        
#         if target_idx in self.cont_indices:
#             return F.l1_loss(preds[mask, target_idx], targets[mask, target_idx])
#         elif target_idx in self.per_angle_indices:
#             pred_rad = torch.deg2rad(preds[mask, target_idx])
#             true_rad = torch.deg2rad(targets[mask, target_idx])
#             diff = torch.atan2(torch.sin(pred_rad - true_rad), torch.cos(pred_rad - true_rad)).abs()
#             return diff.mean()
#         return torch.tensor(float('nan'), device=preds.device)

# class MixedExplainedVarianceLoss(BaseMetric):
#     def __init__(self, cont_indices, per_indices, per_angle_indices, ignore_value=-10.0):
#         super().__init__()
#         self.cont_indices = cont_indices
#         self.per_indices = per_indices
#         self.per_angle_indices = per_angle_indices
#         self.ignore_value = ignore_value

#     def forward(self, preds, targets):
#         if preds.ndim == 1:
#             preds = preds.unsqueeze(1)
#             targets = targets.unsqueeze(1)
#         total_var = 0.0
#         count = 0

#         # Continuous targets
#         for idx in self.cont_indices:
#             mask = targets[:, idx] != self.ignore_value
#             if mask.any():
#                 y = targets[mask, idx]
#                 y_pred = preds[mask, idx]
#                 var_y = torch.var(y)
#                 var_residual = torch.var(y - y_pred)
#                 explained_var = 1 - var_residual / var_y if var_y > 0 else torch.tensor(0.0, device=preds.device)
#                 total_var += explained_var
#                 count += 1

#         # Periodic targets (sin/cos pair form)
#         for i in range(0, len(self.per_indices), 2):
#             sin_idx = self.per_indices[i]
#             cos_idx = self.per_indices[i + 1]
#             mask = (targets[:, sin_idx] != self.ignore_value) & (targets[:, cos_idx] != self.ignore_value)
#             if mask.any():
#                 true_theta = torch.atan2(targets[mask, sin_idx], targets[mask, cos_idx])
#                 pred_theta = torch.atan2(preds[mask, sin_idx], preds[mask, cos_idx])
#                 diff = torch.atan2(torch.sin(pred_theta - true_theta), torch.cos(pred_theta - true_theta))
#                 var_diff = torch.var(diff)
#                 var_true = torch.var(true_theta)
#                 explained_var = 1 - var_diff / var_true if var_true > 0 else torch.tensor(0.0, device=preds.device)
#                 total_var += explained_var
#                 count += 1

#         if count == 0:
#             return torch.tensor(float('nan'), device=preds.device)
#         return total_var / count

#     def forward_test(self, preds, targets):
#         if preds.ndim == 1:
#             preds = preds.unsqueeze(1)
#             targets = targets.unsqueeze(1)
#         total_var = 0.0
#         count = 0

#         for idx in self.cont_indices:
#             mask = targets[:, idx] != self.ignore_value
#             if mask.any():
#                 y = targets[mask, idx]
#                 y_pred = preds[mask, idx]
#                 var_y = torch.var(y)
#                 var_residual = torch.var(y - y_pred)
#                 explained_var = 1 - var_residual / var_y if var_y > 0 else torch.tensor(0.0, device=preds.device)
#                 total_var += explained_var
#                 count += 1

#         for idx in self.per_angle_indices:
#             mask = targets[:, idx] != self.ignore_value
#             if mask.any():
#                 y = torch.deg2rad(targets[mask, idx])
#                 y_pred = torch.deg2rad(preds[mask, idx])
#                 diff = torch.atan2(torch.sin(y_pred - y), torch.cos(y_pred - y))
#                 var_diff = torch.var(diff)
#                 var_y = torch.var(y)
#                 explained_var = 1 - var_diff / var_y if var_y > 0 else torch.tensor(0.0, device=preds.device)
#                 total_var += explained_var
#                 count += 1

#         if count == 0:
#             return torch.tensor(float('nan'), device=preds.device)
#         return total_var / count

#     def forward_test_target(self, preds, targets, target_idx):
#         if preds.ndim == 1:
#             preds = preds.unsqueeze(1)
#             targets = targets.unsqueeze(1)

#         mask = targets[:, target_idx] != self.ignore_value
#         if not mask.any():
#             return torch.tensor(float('nan'), device=preds.device)

#         y = targets[mask, target_idx]
#         y_pred = preds[mask, target_idx]

#         if target_idx in self.cont_indices:
#             var_y = torch.var(y)
#             var_residual = torch.var(y - y_pred)
#             return 1 - var_residual / var_y if var_y > 0 else torch.tensor(0.0, device=preds.device)

#         elif target_idx in self.per_angle_indices:
#             y = torch.deg2rad(y)
#             y_pred = torch.deg2rad(y_pred)
#             diff = torch.atan2(torch.sin(y_pred - y), torch.cos(y_pred - y))
#             var_diff = torch.var(diff)
#             var_y = torch.var(y)
#             return 1 - var_diff / var_y if var_y > 0 else torch.tensor(0.0, device=preds.device)

#         raise RuntimeError("Target index was not in cont or periodic target lists.")