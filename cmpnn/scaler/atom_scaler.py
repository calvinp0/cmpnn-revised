# selective_atom_descriptor_scaler.py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler  # or any sklearn scaler

class AtomDescriptorScaler:
    """
    Scale only selected columns within the 'extra' descriptor slice of f_atoms.
    Optionally mirror the (updated) atom features back into the atom-prefix of f_bonds.
    If mirror_into_bonds is None, auto-detect by checking if f_bonds' prefix equals f_atoms[b2a].
    """

    def __init__(self, atom_base_dim, extra_dim,
                 scale_idx=None,                  # <— indices relative to the extra slice
                 scaler_cls=StandardScaler,
                 mirror_into_bonds=None):
        self.atom_base_dim = int(atom_base_dim)
        self.extra_dim = int(extra_dim)
        if scale_idx is None:
            self.scale_idx = list(range(self.extra_dim))
        else:
            # dedupe, cast to int, keep only valid indices
            idx = {int(i) for i in scale_idx if 0 <= int(i) < self.extra_dim}
            if not idx and self.extra_dim > 0:
                raise ValueError("scale_idx has no valid indices within [0, extra_dim).")
            self.scale_idx = sorted(idx)
        self.scalers = [scaler_cls() for _ in self.scale_idx]
        self._mirror_flag = mirror_into_bonds  # None => auto-detect in fit
        self.fitted = False

    def _auto_detect_mirror(self, mol) -> bool:
        A_raw = getattr(mol, "f_atoms_raw", getattr(mol, "f_atoms", None))
        B_raw = getattr(mol, "f_bonds_raw", getattr(mol, "f_bonds", None))
        b2a   = getattr(mol, "b2a", None)
        if A_raw is None or B_raw is None or b2a is None:
            return False
        atom_dim = A_raw.shape[1]
        if B_raw.shape[1] < atom_dim:
            return False
        src = torch.as_tensor(b2a, dtype=torch.long, device=A_raw.device)
        return torch.allclose(B_raw[:, :atom_dim], A_raw.index_select(0, src))

    def fit(self, mols):
        # decide mirroring
        if self._mirror_flag is None:
            self._mirror_flag = any(self._auto_detect_mirror(m) for m in mols)

        # nothing to scale
        if self.extra_dim == 0 or len(self.scale_idx) == 0:
            self.fitted = True
            return

        # gather per-column data from *_raw if present
        base = self.atom_base_dim
        cols_global = [base + i for i in self.scale_idx]
        per_col = [[] for _ in self.scale_idx]

        for m in mols:
            A_raw = getattr(m, "f_atoms_raw", getattr(m, "f_atoms", None))
            if A_raw is None:
                continue
            for j, c in enumerate(cols_global):
                # keep as (n,1) to please sklearn
                per_col[j].append(A_raw[:, c:c+1].detach().cpu().numpy())

        # fit each scaler independently
        for j, parts in enumerate(per_col):
            if parts:
                Xj = np.vstack(parts)  # (sum_atoms, 1)
                self.scalers[j].fit(Xj)

        self.fitted = True

    @torch.no_grad()
    def transform(self, mols):
        assert self.fitted, "Call fit(...) before transform(...)."

        base = self.atom_base_dim
        cols_global = [base + i for i in self.scale_idx]

        for m in mols:
            A_raw = getattr(m, "f_atoms_raw", getattr(m, "f_atoms", None))
            if A_raw is None:
                continue

            # start from RAW to stay idempotent
            A = A_raw.clone()

            # scale only the selected columns
            for j, c in enumerate(cols_global):
                col = A_raw[:, c:c+1].detach().cpu().numpy()       # (n,1)
                col_t = self.scalers[j].transform(col)             # (n,1)
                A[:, c:c+1] = torch.from_numpy(col_t).to(A.device, dtype=A.dtype)

            # write back
            m.f_atoms = A

            # mirror into f_bonds atom-prefix only if we detected/forced mirroring
            if self._mirror_flag:
                B = getattr(m, "f_bonds", None)
                b2a = getattr(m, "b2a", None)
                if B is not None and b2a is not None and B.shape[1] >= A.shape[1]:
                    src = torch.as_tensor(b2a, dtype=torch.long, device=A.device)
                    B[:, :A.shape[1]] = A.index_select(0, src)
