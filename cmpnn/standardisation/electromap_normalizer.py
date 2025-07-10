import torch

class ElectromapNormalizer:
    """
    Normalizes R and A atom-level features using their masks.
    Assumes atom feature vectors end with the following 7 elements:
    [R, A, sin(D), cos(D), R_mask, A_mask, D_mask]
    where R, A, and D are the atom-level features for radius, angle, and dihedral respectively.
    """

    def __init__(self, r_idx=-7, a_idx=-6, r_mask_idx=-3, a_mask_idx=-2):
        self.r_idx = r_idx
        self.a_idx = a_idx
        self.r_mask_idx = r_mask_idx
        self.a_mask_idx = a_mask_idx
        self.r_mean = None
        self.r_std = None
        self.a_mean = None
        self.a_std = None
        self.fitted = False

    def fit(self, data_list):
        """
        data_list: List of MoleculeData or List[List[MoleculeData]]
        """
        all_atoms = []

        for entry in data_list:
            mols = entry if isinstance(entry, list) else [entry]
            for mol in mols:
                all_atoms.append(mol.f_atoms)
        
        atoms = torch.cat(all_atoms, dim=0)

        r_vals = atoms[:, self.r_idx]
        a_vals = atoms[:, self.a_idx]
        r_mask = atoms[:, self.r_mask_idx]
        a_mask = atoms[:, self.a_mask_idx]

        self.r_mean = r_vals[r_mask==1.0].mean()
        self.r_std = r_vals[r_mask==1.0].std()
        self.a_mean = a_vals[a_mask==1.0].mean()
        self.a_std = a_vals[a_mask==1.0].std()
        self.fitted = True

    def transform(self, f_atoms: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()")
        f = f_atoms.clone()
        r_mask, a_mask = f[:, self.r_mask_idx], f[:, self.a_mask_idx]
        f[r_mask == 1, self.r_idx] = (f[r_mask == 1, self.r_idx] - self.r_mean) / self.r_std
        f[a_mask == 1, self.a_idx] = (f[a_mask == 1, self.a_idx] - self.a_mean) / self.a_std
        return f

    def inverse_transform(self, f_atoms: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Call fit() before inverse_transform()")
        f = f_atoms.clone()
        r_mask, a_mask = f[:, self.r_mask_idx], f[:, self.a_mask_idx]
        f[r_mask == 1, self.r_idx] = f[r_mask == 1, self.r_idx] * self.r_std + self.r_mean
        f[a_mask == 1, self.a_idx] = f[a_mask == 1, self.a_idx] * self.a_std + self.a_mean
        return f