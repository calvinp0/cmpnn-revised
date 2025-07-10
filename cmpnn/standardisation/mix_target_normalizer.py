from typing import List, Tuple, Union
import torch


class BaseNormalizer:
    def __init__(self, ignore_value: float = -10.0):
        self.ignore_value = ignore_value
        self.mean = None
        self.mu = None
        self.std = None

    def fit(self, y: torch.Tensor):
        mask = y != self.ignore_value
        self.mean = y[mask].mean()
        self.std  = y[mask].std()
        # aliases for code that expects mu / std
        self.mu   = self.mean

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return torch.where(
            y == self.ignore_value,
            y,
            (y - self.mean) / self.std,
        )
    
    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        return torch.where(
            y == self.ignore_value,
            y,
            y * self.std + self.mean,
        )
    

class LogNormalizer(BaseNormalizer):
    def __init__(self, shift: float, ignore_value: float = -10.0):
        super().__init__(ignore_value)
        self.shift = shift
    
    def fit(self, y: torch.Tensor):
        y = torch.log(y - self.shift)
        super().fit(y)

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        y = torch.where(y == self.ignore_value, y, torch.log(y - self.shift))
        return super().transform(y)

    def inverse_transform(self, y):
        y = super().inverse_transform(y)
        return torch.where(y == self.ignore_value, y, torch.exp(y) + self.shift)
    

class PeriodicNormalizer:
    def __init__(self, ignore_value: float = -10.0):
        self.ignore_value = ignore_value

    @property
    def expansion_width(self) -> int:
        return 2

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        encoded = []
        for val in y:
            if val == self.ignore_value:
                encoded.append(torch.tensor([self.ignore_value, self.ignore_value]))
            else:
                rad = torch.deg2rad(val)
                encoded.append(torch.tensor([torch.sin(rad), torch.cos(rad)]))
        return torch.cat(encoded)

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        sin_val, cos_val = y[0], y[1]
        if sin_val == self.ignore_value or cos_val == self.ignore_value:
            return torch.tensor([self.ignore_value])
        return torch.tensor([self.robust_inverse_angle(sin_val, cos_val)])

    def inverse_transform_batch(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (N, 2) tensor of sin and cos columns
        returns: (N,) tensor of angles
        """
        sin_col = y[:, 0]
        cos_col = y[:, 1]
        angle = self.robust_inverse_angle(sin_col, cos_col)
        ignore_mask = (sin_col == self.ignore_value) | (cos_col == self.ignore_value)
        angle[ignore_mask] = self.ignore_value
        return angle
    
    @staticmethod
    def robust_inverse_angle(sin_val: torch.Tensor, cos_val: torch.Tensor, eps: float = 1e-8):
        norm = torch.sqrt(sin_val**2 + cos_val**2 + eps)
        normed_sin = sin_val / norm
        normed_cos = cos_val / norm
        theta_rad = torch.atan2(normed_sin, normed_cos)
        return torch.rad2deg(theta_rad) % 360

class MixedTargetNormalizer:
    def __init__(self, kinds: List[str], normalizers: List[BaseNormalizer], ignore_value: float = -10.0):
        assert len(kinds) == len(normalizers), "Length of kinds and normalizers must match"
        assert all(k in ["continuous", "log", "periodic"] for k in kinds), "Invalid kind specified"
        self.kinds = kinds
        self.ignore_value = ignore_value
        self.normalizers = normalizers
        self.periodic_normalizer = PeriodicNormalizer(ignore_value=self.ignore_value)


    def fit(self, ys: torch.Tensor):
        idx = 0
        mus, stds = [] , []
        for i, kind in enumerate(self.kinds):
            if kind in ("continuous", "log"):
                self.normalizers[i].fit(ys[:, idx])
                mus.append(self.normalizers[i].mu)     # 1‑D tensors
                stds.append(self.normalizers[i].std)
                idx += 1
            elif kind == "periodic":
                mus.append(None)
                stds.append(None)
                idx += 1

        # flatten to lists so callers can do normalizer.mu[col]
        self.mu  = mus
        self.std = stds
    
    def transform(self, y: torch.Tensor) -> torch.Tensor:
        out = []
        idx = 0
        for i, kind in enumerate(self.kinds):
            if kind == "continuous" or kind == "log":
                out.append(self.normalizers[i].transform(torch.tensor([y[idx]])))
                idx +=1
            elif kind == "periodic":
                 out.append(self.periodic_normalizer.transform(torch.tensor([y[idx]])))
                 idx += 1
        return torch.cat(out)
    
    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """
        Handles both single-vector (1D) and batched (2D) inverse transformation.
        """
        if y.dim() == 1:
            return self._inverse_transform_single(y)
        elif y.dim() == 2:
            return self._inverse_transform_batch(y)
        else:
            raise ValueError(f"Unexpected tensor shape: {y.shape}")

    def inverse_transform_single_target(self, y: torch.Tensor, target_idx: int) -> torch.Tensor:
        """
        Only inverse-transforms a single target type (used in focus mode).
        Assumes y is shape [N, 1] or [N, 2] depending on kind.
        """
        kind = self.kinds[target_idx]
        if kind in ["continuous", "log"]:
            return self.normalizers[target_idx].inverse_transform(y)
        elif kind == "periodic":
            return self.periodic_normalizer.inverse_transform_batch(y)
        else:
            raise ValueError(f"Unknown target kind: {kind}")

    def _inverse_transform_single(self, y: torch.Tensor) -> torch.Tensor:
        out = []
        i = 0
        j = 0
        while i < len(self.kinds):
            if self.kinds[i] in ["continuous", "log"]:
                out.append(self.normalizers[i].inverse_transform(y[j: j + 1]))
                j += 1
            elif self.kinds[i] == "periodic":
                out.append(self.periodic_normalizer.inverse_transform(y[j:j + 2]))
                j += 2
            i += 1
        return torch.cat(out)

    def _inverse_transform_batch(self, ys: torch.Tensor) -> torch.Tensor:
        N = ys.shape[0]
        out = []
        j = 0
        for i, kind in enumerate(self.kinds):
            if kind in ["continuous", "log"]:
                col = ys[:, j]
                mask = col == self.ignore_value
                val = self.normalizers[i].inverse_transform(col)
                val[mask] = self.ignore_value
                out.append(val.unsqueeze(1))
                j += 1
            elif kind == "periodic":
                pair = ys[:, j:j+2]
                angle = self.periodic_normalizer.inverse_transform_batch(pair)
                out.append(angle.unsqueeze(1))
                j += 2
        return torch.cat(out, dim=1)

    @property
    def mean_(self):        # sklearn-style alias
        return self.mu

    @property
    def scale_(self):
        return self.std


# class MixedTargetNormalizer:
#     def __init__(self, kinds: List[str], ignore_value: float = -10.0):
#         self.kinds = kinds
#         self.ignore_value = ignore_value
#         self.standardizer = TargetNormalizer(ignore_value=ignore_value)
#         self.cont_indices = [i for i, kind in enumerate(kinds) if kind == "continuous"]
#         self.per_indices = [i for i, kind in enumerate(kinds) if kind == "periodic"]

#     def fit(self, ys: torch.Tensor):
#         if ys.ndim == 1:
#             ys = ys.unsqueeze(1)
#         std_targets = ys[:, self.cont_indices]
#         self.standardizer.fit(std_targets)

#     def transform(self, y: torch.Tensor) -> torch.Tensor:
#         # Split into continuous and periodic values
#         cont = y[self.cont_indices]
#         per = y[self.per_indices]

#         normed_cont = self.standardizer.transform(cont)

#         encoded_per = []
#         for val in per:
#             if val == self.ignore_value:
#                 encoded = torch.tensor([self.ignore_value, self.ignore_value])
#             else:
#                 rad = torch.deg2rad(val)
#                 encoded = torch.tensor([torch.sin(rad), torch.cos(rad)])
#             encoded_per.append(encoded)

#         return torch.cat([normed_cont] + encoded_per)

#     def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:

#         y = y.detach().cpu()

#         # Handle single-target debug mode
#         if len(y) < len(self.kinds):
#             # Figure out what kind of target we're dealing with
#             if len(y) == 1:
#                 # Could be continuous or invalid periodic
#                 if self.cont_indices == [0]:  # assume it's cont target at index 0
#                     return self.standardizer.inverse_transform(y.unsqueeze(0)).squeeze(0)
#                 else:
#                     return y  # just return it as-is
#             elif len(y) == 2:
#                 # Could be sin/cos for a periodic target
#                 sin_val, cos_val = y[0], y[1]
#                 if sin_val == self.ignore_value or cos_val == self.ignore_value:
#                     return torch.tensor([self.ignore_value])
#                 else:
#                     theta_rad = torch.atan2(sin_val, cos_val)
#                     theta = torch.rad2deg(theta_rad) % 360
#                     return torch.tensor([theta])
#             else:
#                 raise ValueError(f"Unexpected vector length in debug inverse_transform: {len(y)}")

#         out = []
#         j = 0
#         cont_vals = []
#         ignore_mask = []

#         for kind in self.kinds:
#             if kind == "periodic":
#                 sin_val = y[j]
#                 cos_val = y[j + 1]
#                 if sin_val == self.ignore_value or cos_val == self.ignore_value:
#                     theta = torch.tensor(self.ignore_value)
#                 else:
#                     theta_rad = torch.atan2(sin_val, cos_val)
#                     theta = torch.rad2deg(theta_rad) % 360
#                 out.append(theta.unsqueeze(0))
#                 j += 2

#             else:  # continuous
#                 val = y[j]
#                 cont_vals.append(val)
#                 ignore_mask.append(val == self.ignore_value)
#                 j += 1

#         cont_tensor = torch.stack(cont_vals).unsqueeze(0)  # shape [1, num_cont]
#         inverse_cont = self.standardizer.inverse_transform(cont_tensor).squeeze(0)  # shape [num_cont]

#         # Replace ignored indices
#         for i, is_ignored in enumerate(ignore_mask):
#             if is_ignored:
#                 inverse_cont[i] = self.ignore_value

#         # Insert inversed continuous values in order
#         cont_tensors = [inverse_cont[i].unsqueeze(0) for i in range(len(inverse_cont))]
#         out = cont_tensors + out  # continuous first (match original order)

#         return torch.cat(out)

