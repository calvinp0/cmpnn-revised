import numpy as np
from typing import List, Tuple, Sequence, Optional, Union
from sklearn.metrics import pairwise_distances
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from cmpnn.split.base import BaseSplitter

ArrayLikeMol = Union[str, Chem.Mol]


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def generalized_tanimoto_square(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    G = X @ X.T
    sq = np.einsum("ij,ij->i", X, X)
    denom = sq[:, None] + sq[None, :] - G
    sim = _safe_div(G, denom)
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    D[~np.isfinite(D)] = 0.0
    D[D < 0] = 0.0
    return 0.5 * (D + D.T)


def generalized_tanimoto_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    G = A @ B.T
    sqA = np.einsum("ij,ij->i", A, A)
    sqB = np.einsum("ij,ij->i", B, B)
    denom = sqA[:, None] + sqB[None, :] - G
    sim = _safe_div(G, denom)
    D = 1.0 - sim
    D[~np.isfinite(D)] = 0.0
    D[D < 0] = 0.0
    return D


def order_invariant_two_component(
    D_AA: np.ndarray, D_BB: np.ndarray, D_AB: np.ndarray
) -> np.ndarray:
    aligned = D_AA + D_BB
    swapped = D_AB + D_AB.T
    D = np.minimum(aligned, swapped)
    D[~np.isfinite(D)] = 0.0
    D[D < 0] = 0.0
    return 0.5 * (D + D.T)


def aggregate_joint(
    D1: np.ndarray, D2: np.ndarray, mode: str, w: float = 0.5, p: float = 2.0
) -> np.ndarray:
    if D1.shape != D2.shape:
        raise ValueError("Joint aggregation requires matching shapes.")
    if mode == "mean":
        return w * D1 + (1.0 - w) * D2
    if mode == "max":
        return np.maximum(D1, D2)
    if mode == "p-norm":
        return ((w * (D1**p)) + ((1.0 - w) * (D2**p))) ** (1.0 / p)
    raise ValueError(f"Unknown joint mode: {mode}")


def kennard_stone_order(D: np.ndarray, seed: int = 42) -> np.ndarray:
    D = np.array(D, dtype=float, copy=True)
    D = np.nan_to_num(D, posinf=np.inf, neginf=np.inf)
    D[D < 0] = 0.0
    D = 0.5 * (D + D.T)
    n = D.shape[0]
    np.fill_diagonal(D, -np.inf)

    i1, i2 = np.unravel_index(np.nanargmax(D), D.shape)
    order = np.full(n, -1, dtype=int)
    order[0], order[1] = i1, i2

    selected = np.zeros(n, dtype=bool)
    selected[[i1, i2]] = True
    min_dist = np.minimum(D[:, i1], D[:, i2])
    min_dist[selected] = -np.inf

    for t in range(2, n):
        m = np.max(min_dist)
        next_idx = int(np.flatnonzero(min_dist == m)[0])
        order[t] = next_idx
        selected[next_idx] = True
        np.minimum(min_dist, D[:, next_idx], out=min_dist)
        min_dist[selected] = -np.inf
    return order


def _featurize_smiles(smiles: Sequence[ArrayLikeMol], radius: int, n_bits: int) -> np.ndarray:
    X = []
    for smi in smiles:
        if isinstance(smi, Chem.Mol):
            mol = smi
        else:
            mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid molecule: {smi!r}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=float)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(arr)
    return np.asarray(X, dtype=float)


class KennardStoneSplitter(BaseSplitter):
    def __init__(
        self,
        distance_metric: str = "jaccard",
        joint_mode: str = "mean",
        donor_weight: float = 0.5,
        p_norm: float = 2.0,
        radius: int = 2,
        n_bits: int = 2048,
        seed: int = 42,
    ):
        super().__init__(seed)
        self.distance_metric = (
            "jaccard" if distance_metric in ("tanimoto", "jaccard") else distance_metric
        )
        self.joint_mode = joint_mode.replace("_", "-")
        self.donor_weight = donor_weight
        self.p_norm = p_norm
        self.radius = radius
        self.n_bits = n_bits

    def _build_D(self, X: np.ndarray) -> np.ndarray:
        if self.distance_metric == "jaccard":
            return generalized_tanimoto_square(X)
        D = pairwise_distances(X, metric=self.distance_metric)
        D[np.isnan(D)] = 0.0
        D[D < 0] = 0.0
        return 0.5 * (D + D.T)

    def _build_cross(self, XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
        if self.distance_metric == "jaccard":
            return generalized_tanimoto_cross(XA, XB)
        D = pairwise_distances(XA, XB, metric=self.distance_metric)
        D[np.isnan(D)] = 0.0
        D[D < 0] = 0.0
        return D

    def _compute_D_single(self, smiles: Sequence[ArrayLikeMol]) -> np.ndarray:
        X = _featurize_smiles(smiles, self.radius, self.n_bits)
        return self._build_D(X)

    def _compute_D_pairs(
        self, donors: Sequence[ArrayLikeMol], acceptors: Sequence[ArrayLikeMol]
    ) -> np.ndarray:
        X_donors = _featurize_smiles(donors, self.radius, self.n_bits)
        X_acceptors = _featurize_smiles(acceptors, self.radius, self.n_bits)

        if self.joint_mode == "concat":
            X = np.concatenate((X_donors, X_acceptors), axis=1)
            return self._build_D(X)
        if self.joint_mode == "order-invariant":
            D_AA = self._build_D(X_donors)
            D_BB = self._build_D(X_acceptors)
            D_AB = self._build_cross(X_donors, X_acceptors)
            return order_invariant_two_component(D_AA, D_BB, D_AB)

        D1 = self._build_D(X_donors)
        D2 = self._build_D(X_acceptors)
        return aggregate_joint(
            D1,
            D2,
            mode=self.joint_mode,
            w=self.donor_weight,
            p=self.p_norm,
        )

    def split(
        self,
        dataset,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        return_indices: bool = False,
    ) -> Tuple[List, List, List]:
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

        n_total = len(dataset)
        if n_total == 0:
            return [], [], []

        first = dataset[0]
        if isinstance(first, (tuple, list)):
            donors = [
                getattr(dataset[i][0], "smiles", dataset[i][0]) for i in range(n_total)
            ]
            acceptors = [
                getattr(dataset[i][1], "smiles", dataset[i][1]) for i in range(n_total)
            ]
            D = self._compute_D_pairs(donors, acceptors)
        else:
            smiles = [
                getattr(dataset[i], "smiles", dataset[i]) for i in range(n_total)
            ]
            D = self._compute_D_single(smiles)

        order = kennard_stone_order(D, seed=self.seed)

        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)
        train_idx = order[:n_train]
        val_idx = order[n_train:n_train + n_val]
        test_idx = order[n_train + n_val:]

        if return_indices:
            return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

        def gather(idx_list):
            return [dataset[i] for i in idx_list]

        return gather(train_idx), gather(val_idx), gather(test_idx)
