from typing import Optional, List
import torch
import torch.nn as nn
import numpy as np

try:
    from sklearn.preprocessing import (
        StandardScaler,
        RobustScaler,
        MinMaxScaler,
        PowerTransformer,
    )
except ImportError:
    StandardScaler = RobustScaler = MinMaxScaler = PowerTransformer = tuple()


def _is_standard(t):
    return (
        (
            (StandardScaler is not None and isinstance(t, StandardScaler))
            or (t.__class__.__name__ == "StandardScaler")
        )
        and hasattr(t, "mean_")
        and hasattr(t, "scale_")
    )


def _is_minmax(t):
    return (
        (
            (MinMaxScaler is not None and isinstance(t, MinMaxScaler))
            or (t.__class__.__name__ == "MinMaxScaler")
        )
        and hasattr(t, "data_min_")
        and hasattr(t, "data_max_")
        and hasattr(t, "feature_range")
    )


def _is_robust(t):
    return (
        (RobustScaler is not None and isinstance(t, RobustScaler))
        or (t.__class__.__name__ == "RobustScaler")
    ) and hasattr(t, "scale_")


def _is_power(t):
    return (
        (PowerTransformer is not None and isinstance(t, PowerTransformer))
        or (t.__class__.__name__ == "PowerTransformer")
    ) and hasattr(t, "lambdas_")


class _AffineColumnUnscaler(nn.Module):
    """
    Torch-native inverse for affine scalers: y = (y_scale - mean_) / scale_ (forward)
    Inverse: y = y * scale_ + mean_
    Handles per-column params as registered buffers so it checkpoints cleanly.
    """

    def __init__(
        self, mean: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None
    ):
        super().__init__()
        # mean/scale can be None for scalers that center or scale selectively; use zeros/ones
        if mean is None:
            mean = torch.zeros(1)
        if scale is None:
            scale = torch.ones(1)
        self.register_buffer("mean", mean)  # [n_cols]
        self.register_buffer("scale", scale)  # [n_cols]

    def forward(self, y_scaled: torch.Tensor) -> torch.Tensor:
        # y_scaled: [B, T]
        if self.mean.numel() == 0 and self.scale.numel() == 0:
            return y_scaled
        return y_scaled * self.scale + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, scale={self.scale})"


class ColumnUnscaler(nn.Module):
    """
    Wrap a list of sklearn-style transformers (one per target column).
    For known affine scalers (Standard/MinMax/Robust) -> vectorized Torch path.
    Otherwise -> safe NumPy fallback (CPU); still works during eval/metric logging.
    """

    def __init__(self, transformers):
        super().__init__()
        self.transformers = list(transformers)

        a_list, b_list, aff_mask = [], [], []
        pt_mask, pt_method, pt_lambda, pt_std, pt_mu, pt_sigma = [], [], [], [], [], []

        for t in self.transformers:
            if _is_standard(t):
                a_list.append(float(np.ravel(t.scale_)[0]))
                b_list.append(float(np.ravel(t.mean_)[0]))
                aff_mask.append(True)
                pt_mask.append(False)
                pt_method.append("")
                pt_lambda.append(0.0)
                pt_std.append(False)
                pt_mu.append(0.0)
                pt_sigma.append(1.0)

            elif _is_minmax(t):
                dmin = float(np.ravel(t.data_min_)[0])
                dmax = float(np.ravel(t.data_max_)[0])
                fr_min, fr_max = t.feature_range
                a = (dmax - dmin) / (fr_max - fr_min)
                b = dmin - a * fr_min
                a_list.append(a)
                b_list.append(b)
                aff_mask.append(True)
                pt_mask.append(False)
                pt_method.append("")
                pt_lambda.append(0.0)
                pt_std.append(False)
                pt_mu.append(0.0)
                pt_sigma.append(1.0)

            elif _is_robust(t):
                center = float(np.ravel(getattr(t, "center_", [0.0]))[0])
                scale = float(np.ravel(getattr(t, "scale_", [1.0]))[0])
                a_list.append(scale)
                b_list.append(center)
                aff_mask.append(True)
                pt_mask.append(False)
                pt_method.append("")
                pt_lambda.append(0.0)
                pt_std.append(False)
                pt_mu.append(0.0)
                pt_sigma.append(1.0)

            elif _is_power(t):
                a_list.append(1.0)
                b_list.append(0.0)
                aff_mask.append(False)
                pt_mask.append(True)
                pt_method.append(t.method)  # 'yeo-johnson' or 'box-cox'
                pt_lambda.append(float(np.ravel(t.lambdas_)[0]))
                stdize = bool(getattr(t, "standardize", True))
                pt_std.append(stdize)
                if stdize and hasattr(t, "_scaler"):
                    pt_mu.append(float(np.ravel(t._scaler.mean_)[0]))
                    pt_sigma.append(float(np.ravel(t._scaler.scale_)[0]))
                else:
                    pt_mu.append(0.0)
                    pt_sigma.append(1.0)

            else:
                a_list.append(1.0)
                b_list.append(0.0)
                aff_mask.append(False)
                pt_mask.append(False)
                pt_method.append("")
                pt_lambda.append(0.0)
                pt_std.append(False)
                pt_mu.append(0.0)
                pt_sigma.append(1.0)

        # buffers
        self.register_buffer("aff_a", torch.tensor(a_list, dtype=torch.float32))
        self.register_buffer("aff_b", torch.tensor(b_list, dtype=torch.float32))
        self.register_buffer("aff_mask", torch.tensor(aff_mask, dtype=torch.bool))

        self.pt_method = pt_method
        self.register_buffer("pt_mask", torch.tensor(pt_mask, dtype=torch.bool))
        self.register_buffer("pt_lambda", torch.tensor(pt_lambda, dtype=torch.float32))
        self.register_buffer("pt_std", torch.tensor(pt_std, dtype=torch.bool))
        self.register_buffer("pt_mu", torch.tensor(pt_mu, dtype=torch.float32))
        self.register_buffer("pt_sigma", torch.tensor(pt_sigma, dtype=torch.float32))

    @staticmethod
    def _inv_box_cox(z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Inverse Box-Cox transform."""
        # z, lam broadcast on last dim
        eps = 1e-12
        out = torch.where(
            torch.abs(lam) > 1e-12,
            torch.pow(lam * z + 1.0, 1.0 / (lam + eps)),
            torch.exp(z),
        )
        return out

    @staticmethod
    def _inv_yeo_johnson(z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Inverse Yeo-Johnson transform.
        Split by sign of original x; YJ forward def implies:
        - for x >= 0: z = (x + 1) ** lam - 1 / lam if lam != 0; log(x + 1) if lam == 0
        - for x < 0: z = -(1 - x) ** (2 - lam) / (2 - lam) if lam != 2; -log(1 - x) if lam == 2
        Inverse requires choosing branch. The sign of original x corresponds to :
        if z >= 0 -> x >= 0 branch; if z < 0 -> x < 0 branch (property of YJ).
        """
        eps = 1e-12
        pos_branch = z >= 0
        lam_ne0 = torch.abs(lam) > eps
        lam_ne2 = torch.abs(lam - 2.0) > eps

        x_pos = torch.where(
            lam_ne0,
            torch.pow(lam * z + 1.0, 1.0 / (lam + eps)) - 1.0,
            torch.exp(z) - 1.0,
        )
        x_neg = torch.where(
            lam_ne2,
            1.0 - torch.pow(-(2.0 - lam) * z + 1.0, 1.0 / (2.0 - lam + eps)),
            1.0 - torch.exp(-z),
        )
        return torch.where(pos_branch, x_pos, x_neg)

    def forward(self, y_scaled: torch.Tensor) -> torch.Tensor:
        y = y_scaled.clone()
        dev = y.device
        eps = 1e-12

        # 1) Affine columns: x = a*z + b
        if self.aff_mask.any():
            m = self.aff_mask.to(dev)
            y[:, m] = y[:, m] * self.aff_a[m].to(dev) + self.aff_b[m].to(dev)

        # 2) PowerTransformer columns (destandardize -> inverse power)
        if self.pt_mask.any():
            cols = self.pt_mask.nonzero(as_tuple=False).view(-1).tolist()
            for col in cols:
                z = y[:, col]
                if bool(self.pt_std[col].item()):
                    z = z * self.pt_sigma[col].to(dev) + self.pt_mu[col].to(dev)
                lam = self.pt_lambda[col].to(dev)

                if self.pt_method[col] == "box-cox":
                    x = torch.where(
                        torch.abs(lam) > eps,
                        torch.pow(lam * z + 1.0, 1.0 / (lam + eps)),
                        torch.exp(z),
                    )
                else:  # yeo-johnson
                    pos = z >= 0
                    lam_ne0 = torch.abs(lam) > eps
                    lam_ne2 = torch.abs(lam - 2.0) > eps
                    x_pos = torch.where(
                        lam_ne0,
                        torch.pow(lam * z + 1.0, 1.0 / (lam + eps)) - 1.0,
                        torch.exp(z) - 1.0,
                    )
                    x_neg = torch.where(
                        lam_ne2,
                        1.0
                        - torch.pow(-(2.0 - lam) * z + 1.0, 1.0 / (2.0 - lam + eps)),
                        1.0 - torch.exp(-z),
                    )
                    x = torch.where(pos, x_pos, x_neg)

                y[:, col] = x

        # 3) Fallback for any unknowns
        remain = (~self.aff_mask) & (~self.pt_mask)
        if remain.any():
            cols = remain.nonzero(as_tuple=False).view(-1).tolist()
            arr = y[:, cols].detach().cpu().numpy()
            outs = [
                self.transformers[c].inverse_transform(arr[:, [j]])
                for j, c in enumerate(cols)
            ]
            y[:, cols] = torch.from_numpy(np.hstack(outs)).to(y.dtype).to(dev)

        return y
