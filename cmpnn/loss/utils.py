from cmpnn.loss.mixed_loss_old import MixedRMSELoss, MixedMAELoss, MixedR2Score, MixedExplainedVarianceLoss, MixedMSELoss, MixedLossFactory
from cmpnn.loss.masked_loss import MaskedRMSE, MaskedMAE, MaskedR2Score
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score


def get_metric_by_name(name, cont_indices=None, per_indices=None, per_angle_indices=None, ignore_value=-10.0):
    name = name.lower()

    if name in {"hybrid", "mixedmae", "mixedmse", "mixedrmse", "mixedr2", "mixedexplainedvariance"}:
        return MixedLossFactory(
            loss_type=name,
            cont_indices=cont_indices,
            per_indices=per_indices,
            per_angle_indices=per_angle_indices,
            ignore_value=ignore_value,
        )
    elif name == "maskedrmse":
        return MaskedRMSE(ignore_value)
    elif name == "maskedmae":
        return MaskedMAE(ignore_value)
    elif name == "maskedr2":
        return MaskedR2Score(ignore_value)
    elif name == "rmse":
        return MeanSquaredError(squared=False)
    elif name == "mae":
        return MeanAbsoluteError()
    elif name == "r2":
        return R2Score()
    else:
        raise ValueError(f"Unknown metric name: {name}")

def log_gaussian_nll(pred_mu, pred_logvar, target):
    var = pred_logvar.exp()
    return 0.5 * (pred_logvar + ((target - pred_mu) ** 2) / var).mean()
