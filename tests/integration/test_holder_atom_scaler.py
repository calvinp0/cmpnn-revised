import numpy as np
import torch
import pytest

from sklearn.preprocessing import StandardScaler

from cmpnn.data.dataset_holder import MoleculeDatasetHolder, MultiMoleculeDatasetHolder
from cmpnn.data.molecule_data import MoleculeData
from cmpnn.scaler.atom_scaler import AtomDescriptorScaler


# --- tiny helpers -------------------------------------------------------------
class DummyMol(MoleculeData):
    def __init__(self, f_atoms, f_bonds, b2a):
        super().__init__()
        self.f_atoms = torch.tensor(f_atoms, dtype=torch.float32)
        self.f_bonds = torch.tensor(f_bonds, dtype=torch.float32)
        self.b2a = torch.tensor(b2a, dtype=torch.long)
        # also provide minimal globals/targets to keep holder paths happy
        self.global_features = torch.zeros(2, dtype=torch.float32)
        self.y = torch.zeros(1, dtype=torch.float32)

class EchoSplitter:
    """Return the same sequence for train/val/test (keeps indices & ordering)."""
    def split(self, dataset, **_):
        return dataset, list(dataset), list(dataset)

class EchoSplitterMulti:
    def split(self, dataset, **_):
        return dataset, list(dataset), list(dataset)


def _make_mols_with_mirroring(rng, atom_base=4, extra_dim=3, n_mols=5, nA_range=(3,6)):
    mols = []
    for _ in range(n_mols):
        nA = rng.integers(*nA_range)
        base = np.eye(atom_base, dtype=np.float32)[rng.integers(0, atom_base, size=nA)]
        extras = rng.normal(loc=5.0, scale=2.0, size=(nA, extra_dim)).astype(np.float32)
        f_atoms = np.hstack([base, extras])
        atom_dim = f_atoms.shape[1]
        b2a = np.arange(nA, dtype=np.int64)            # one bond per atom (simple mapping)
        f_bonds = np.zeros((nA, atom_dim + 2), dtype=np.float32)  # +2 dummy bond feats
        f_bonds[:, :atom_dim] = f_atoms[b2a]           # mirror atom features into bonds
        mols.append(DummyMol(f_atoms, f_bonds, b2a))
    return mols, atom_base, extra_dim

def _make_mols_without_mirroring(rng, atom_base=4, extra_dim=2, n_mols=4, nA_range=(3,5)):
    mols = []
    for _ in range(n_mols):
        nA = rng.integers(*nA_range)
        base = np.eye(atom_base, dtype=np.float32)[rng.integers(0, atom_base, size=nA)]
        extras = rng.normal(size=(nA, extra_dim)).astype(np.float32)
        f_atoms = np.hstack([base, extras])
        b2a = np.arange(nA, dtype=np.int64)
        f_bonds = np.random.normal(size=(nA, 5)).astype(np.float32)  # no atom prefix here
        mols.append(DummyMol(f_atoms, f_bonds, b2a))
    return mols, atom_base, extra_dim



def test_single_holder_atom_desc_scaling_with_mirroring_and_idempotency():
    rng = np.random.default_rng(0)
    dataset, atom_base, extra_dim = _make_mols_with_mirroring(rng, atom_base=4, extra_dim=3)

    holder = MoleculeDatasetHolder(dataset)
    splitter = EchoSplitter()

    scaler = AtomDescriptorScaler(
        atom_base_dim=atom_base,
        extra_dim=extra_dim,
        scaler_cls=StandardScaler,
        mirror_into_bonds=None,  # auto-detect
    )

    train, val, test = holder.split(
        splitter=splitter,
        atom_desc_scaler=scaler,
        feature_transformer=None,
        target_transformers=None,
    )

    # 1) extras should be ~standardized across all atoms in train
    cols = slice(atom_base, atom_base + extra_dim)
    X = np.vstack([m.f_atoms[:, cols].numpy() for m in train])
    assert np.allclose(X.mean(axis=0), 0.0, atol=0.15)
    assert np.all((X.std(axis=0) > 0.8) & (X.std(axis=0) < 1.2))

    # 2) bonds should mirror atom features after scaling
    for m in train:
        A, B = m.f_atoms, m.f_bonds
        atom_dim = A.shape[1]
        assert torch.allclose(B[:, :atom_dim], A.index_select(0, m.b2a))

    # 3) idempotency: calling transform again should not double-scale
    snap = [m.f_atoms.clone() for m in train]
    scaler.transform(train)
    for a, m in zip(snap, train):
        assert torch.allclose(a, m.f_atoms, atol=1e-7)


def test_single_holder_atom_desc_scaling_without_mirroring():
    rng = np.random.default_rng(1)
    dataset, atom_base, extra_dim = _make_mols_without_mirroring(rng, atom_base=3, extra_dim=2)

    holder = MoleculeDatasetHolder(dataset)
    splitter = EchoSplitter()

    scaler = AtomDescriptorScaler(
        atom_base_dim=atom_base,
        extra_dim=extra_dim,
        scaler_cls=StandardScaler,
        mirror_into_bonds=None,  # auto-detect -> should be False here
    )

    train, _, _ = holder.split(
        splitter=splitter,
        atom_desc_scaler=scaler,
        feature_transformer=None,
        target_transformers=None,
    )

    # extras standardized
    cols = slice(atom_base, atom_base + extra_dim)
    X = np.vstack([m.f_atoms[:, cols].numpy() for m in train])
    assert np.allclose(X.mean(axis=0), 0.0, atol=0.15)

    # bonds NOT mirrored (prefix shouldn't equal atom features)
    for m in train:
        A, B = m.f_atoms, m.f_bonds
        atom_dim = A.shape[1]
        # if shapes mismatch, we're definitely not mirroring
        if B.shape[1] < atom_dim:
            continue
        assert not torch.allclose(B[:, :atom_dim], A.index_select(0, m.b2a))


def test_single_holder_atom_desc_selective_scaling():
    """Scale only a subset of extra columns (e.g., last 2 of 5 extras)."""
    rng = np.random.default_rng(2)
    dataset, atom_base, extra_dim = _make_mols_with_mirroring(rng, atom_base=5, extra_dim=5)

    # choose to scale only indices [2,3,4] within the extra block
    scale_idx = [2, 3, 4]

    holder = MoleculeDatasetHolder(dataset)
    splitter = EchoSplitter()

    scaler = AtomDescriptorScaler(
        atom_base_dim=atom_base,
        extra_dim=extra_dim,
        scaler_cls=StandardScaler,
        mirror_into_bonds=None,
        scale_idx=scale_idx,     # requires your scaler to support scale_idx
    )

    # keep a raw copy for later comparison
    raw_extras = [m.f_atoms[:, atom_base:].clone() for m in dataset]

    train, _, _ = holder.split(
        splitter=splitter,
        atom_desc_scaler=scaler,
        feature_transformer=None,
        target_transformers=None,
    )

    # columns 0..1 in the extra block should be unchanged vs raw; 2..4 standardized
    for m, raw in zip(train, raw_extras):
        after = m.f_atoms[:, atom_base:]
        # unchanged columns
        assert torch.allclose(after[:, :2], raw[:, :2], atol=1e-7)
        # scaled columns ~ zero mean across train atoms
    X_scaled = np.vstack([m.f_atoms[:, atom_base+2:atom_base+5].numpy() for m in train])
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=0.2)


def test_multi_holder_atom_desc_scaling_components_sync():
    """MultiMoleculeDatasetHolder: components should share the same scaled atom extras."""
    rng = np.random.default_rng(3)

    # build paired molecules with the *same* raw features so we can compare directly
    a_list, atom_base, extra_dim = _make_mols_with_mirroring(rng, atom_base=4, extra_dim=3, n_mols=6)
    b_list, _, _ = _make_mols_with_mirroring(rng, atom_base=4, extra_dim=3, n_mols=6)

    # make the second component have identical raw as the first to simplify equality checks
    for a, b in zip(a_list, b_list):
        b.f_atoms = a.f_atoms.clone()
        b.f_bonds = a.f_bonds.clone()
        b.b2a     = a.b2a.clone()

    dataset = [[a, b] for a, b in zip(a_list, b_list)]
    holder = MultiMoleculeDatasetHolder(dataset)
    splitter = EchoSplitterMulti()

    scaler = AtomDescriptorScaler(
        atom_base_dim=atom_base,
        extra_dim=extra_dim,
        scaler_cls=StandardScaler,
        mirror_into_bonds=None,
    )

    train, _, _ = holder.split(
        splitter=splitter,
        atom_desc_scaler=scaler,
        feature_transformers=None,
        target_transformers=None,
    )

    # components have identical scaled atoms and mirrored bonds
    for pair in train:
        A1, A2 = pair[0].f_atoms, pair[1].f_atoms
        B1, B2 = pair[0].f_bonds, pair[1].f_bonds
        assert torch.allclose(A1, A2, atol=1e-7)
        atom_dim = A1.shape[1]
        assert torch.allclose(B1[:, :atom_dim], A1.index_select(0, pair[0].b2a))
        assert torch.allclose(B2[:, :atom_dim], A2.index_select(0, pair[1].b2a))
