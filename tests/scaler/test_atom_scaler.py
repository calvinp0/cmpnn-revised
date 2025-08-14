import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from cmpnn.scaler.atom_scaler import AtomDescriptorScaler


class DummyMol:
    def __init__(self, f_atoms, f_bonds, b2a):
        self.f_atoms = torch.tensor(f_atoms, dtype=torch.float32)
        self.f_bonds = torch.tensor(f_bonds, dtype=torch.float32)
        self.b2a = torch.tensor(b2a, dtype=torch.long)
        self.y = torch.tensor([0.0], dtype=torch.float32)
        self.global_features = torch.tensor([0.0], dtype=torch.float32)


def _ensure_raw(m):
    if not hasattr(m, "f_atoms_raw"):
        m.f_atoms_raw = m.f_atoms.clone()
    if not hasattr(m, "f_bonds_raw"):
        m.f_bonds_raw = m.f_bonds.clone()


def test_atom_extras_scaling_and_mirroring():
    rng = np.random.default_rng(0)
    atom_base = 3
    extra_dim = 2

    mols = []
    for _ in range(4):
        nA = rng.integers(3, 6)
        based = np.eye(atom_base, dtype=np.float32)[rng.integers(0, atom_base, size=nA)]
        extras = rng.normal(5.0, 2.0, size=(nA, extra_dim)).astype(np.float32)
        f_atoms = np.hstack([based, extras])
        atom_dim = f_atoms.shape[1]
        b2a = np.arange(nA, dtype=np.int64)
        f_bonds = np.zeros(
            (nA, atom_dim + 3), dtype=np.float32
        )  # 3 dummy bond features
        f_bonds[:, :atom_dim] = f_atoms[b2a]  # mirror atom features into bonds
        mols.append(DummyMol(f_atoms, f_bonds, b2a))

    for m in mols:
        _ensure_raw(m)
    scaler = AtomDescriptorScaler(
        atom_base_dim=atom_base,
        extra_dim=extra_dim,
        scaler_cls=StandardScaler,
        mirror_into_bonds=None,
    )

    scaler.fit(mols)
    scaler.transform(mols)

    # extra ~ standardised
    cols = slice(atom_base, atom_base + extra_dim)
    X = np.vstack([m.f_atoms[:, cols].numpy() for m in mols])
    assert np.allclose(X.mean(0), 0.0, atol=0.15)
    assert np.all((X.std(0) > 0.8) & (X.std(0) < 1.2))

    # bonds mirrored
    for m in mols:
        A = m.f_atoms
        B = m.f_bonds
        atom_fdim = A.shape[1]
        assert torch.allclose(B[:, :atom_fdim], A.index_select(0, m.b2a), atol=1e-6)

    # idempotent (transform again does not double-scale)
    before_atoms = [m.f_atoms.clone() for m in mols]
    scaler.transform(mols)
    after_atoms  = [m.f_atoms.clone() for m in mols]

    for b, a in zip(before_atoms, after_atoms):
        assert torch.allclose(b, a)


def test_no_mirroring_when_no_prefix():
    rng = np.random.default_rng(1)
    atom_base, extra_dim = 2, 1

    mols = []
    for _ in range(3):
        nA = rng.integers(3, 5)
        base = np.eye(atom_base, dtype=np.float32)[rng.integers(0, atom_base, size=nA)]
        extras = rng.normal(size=(nA, extra_dim)).astype(np.float32)
        f_atoms = np.hstack([base, extras])
        b2a = np.arange(nA, dtype=np.int64)
        f_bonds = rng.normal(size=(nA, 5)).astype(np.float32)  # no atom prefix
        mols.append(DummyMol(f_atoms, f_bonds, b2a))

    for m in mols:
        _ensure_raw(m)

    # keep copies per molecule (shapes differ)
    B_before = [m.f_bonds.clone() for m in mols]

    scaler = AtomDescriptorScaler(atom_base_dim=atom_base, extra_dim=extra_dim,
                                scaler_cls=StandardScaler, mirror_into_bonds=None)
    scaler.fit(mols)
    scaler.transform(mols)

    B_after = [m.f_bonds.clone() for m in mols]
    for Bb, Ba in zip(B_before, B_after):
        assert torch.allclose(Bb, Ba)


def test_selective_scaling_ohe_untouched():
    rng = np.random.default_rng(0)
    base_dim, extra_dim = 10, 5
    scale_idx = [2,3,4]  # only last 3 extra cols
    class M:
        pass

    mols = []
    for _ in range(3):
        nA = rng.integers(4, 7)
        # base OHE (10 classes)
        base = np.eye(base_dim, dtype=np.float32)[rng.integers(0, base_dim, size=nA)]
        # extra: 2 OHE-like, 3 continuous
        extra_ohe = np.eye(2, dtype=np.float32)[rng.integers(0, 2, size=nA)]
        extra_cont = rng.normal(5.0, 2.0, size=(nA, 3)).astype(np.float32)
        f_atoms = np.hstack([base, extra_ohe, extra_cont])

        atom_dim = f_atoms.shape[1]
        b2a = np.arange(nA, dtype=np.int64)
        f_bonds = np.zeros((nA, atom_dim + 2), dtype=np.float32)
        f_bonds[:, :atom_dim] = f_atoms[b2a]

        m = M()
        m.f_atoms = torch.tensor(f_atoms)
        m.f_bonds = torch.tensor(f_bonds)
        m.b2a = torch.tensor(b2a)
        # raw snapshots
        m.f_atoms_raw = m.f_atoms.clone()
        m.f_bonds_raw = m.f_bonds.clone()
        mols.append(m)

    scaler = AtomDescriptorScaler(base_dim, extra_dim, scale_idx=scale_idx)
    scaler.fit(mols)
    scaler.transform(mols)

    # base and first 2 extras unchanged
    for m in mols:
        assert torch.allclose(m.f_atoms[:, :base_dim+2], m.f_atoms_raw[:, :base_dim+2])

    # last 3 extras roughly standardized
    cols = slice(base_dim+2, base_dim+5)
    X = np.vstack([m.f_atoms[:, cols].numpy() for m in mols])
    assert np.allclose(X.mean(0), 0.0, atol=0.2)
    assert np.all((X.std(0) > 0.8) & (X.std(0) < 1.2))

    # mirrored into bonds
    for m in mols:
        A = m.f_atoms
        B = m.f_bonds
        atom_dim = A.shape[1]
        assert torch.allclose(B[:, :atom_dim], A.index_select(0, m.b2a))
