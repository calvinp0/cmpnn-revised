
import torch
import torch.nn as nn

class LogGaussLoss(nn.Module):

    def __init__(self, entropy_weight: float = 0.05, kl_weight: float = 0.01, kl_prior_std: float = 1.0, ignore_value: float = -10.0):
        """
        Log Gaussian loss function for regression tasks.

        Args:
            entropy_weight (float): Weight for the entropy term.
            kl_weight (float): Weight for the KL divergence term.
            kl_prior_std (float): Standard deviation of the prior distribution.
            ignore_value (float): Sentinel value in targets to skip.
        """
        super().__init__()
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
        self.kl_prior_std = kl_prior_std
        self.ignore_value = ignore_value

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the log Gaussian loss.

        Args:
            preds (Tensor): Model predictions.
            targets (Tensor): Ground truth values.

        Returns:
            Tensor: Computed loss.
        """
        # Split into mu and log-variance
        mu, logvar = preds.chunk(2, dim=-1)
        # Mask invalid entries
        mask = (targets != self.ignore_value)
        if not mask.any():
            return torch.tensor(0.0, device=preds.device)
        mu = mu[mask]
        logvar = logvar[mask]
        targets = targets[mask]

        # Convert logvar to variance
        var = logvar.exp()

        # Negative log likelihood = 0.5*(log (var**2) + (target - mu)**2 / var)
        nll = 0.5 * (logvar + (targets - mu).pow(2) / var)
        nll = nll.mean()

        # KL divergence with prior N(0, prior_var)
        prior_var = self.kl_prior_std ** 2
        log_prior_var = torch.log(torch.tensor(prior_var, device=preds.device))
        kl = 0.5 *(
            var /prior_var
            + mu.pow(2) / prior_var
            - 1
            - (logvar - log_prior_var)
        )
        kl = kl.mean()
        
        loss = nll + self.kl_weight * kl

        return loss

    def evaluation(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute evaluation metric. In this case, it's the same as the loss.

        Args:
            preds (Tensor): Model predictions.
            targets (Tensor): Ground truth values.

        Returns:
            Tensor: Computed evaluation metric.
        """
        return self.forward(preds, targets)

    def __repr__(self) -> str:
        """
        String representation of the LogGaussLoss class.
        """
        return f"LogGaussLoss(entropy_weight={self.entropy_weight}, kl_weight={self.kl_weight}, kl_prior_std={self.kl_prior_std})"
    def __str__(self) -> str:
        """
        String representation of the LogGaussLoss class.
        """
        return self.__repr__()
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Call method to compute the loss.
        """
        return self.forward(preds, targets)