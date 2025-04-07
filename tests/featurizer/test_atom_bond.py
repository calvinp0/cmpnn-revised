import pytest
import torch
from rdkit import Chem
from cmpnn.featurizer.atom_bond import AtomFeaturizer, BondFeaturizer


def test_atom_featurizer_shape_and_known_features():
    mol = Chem.MolFromSmiles("c1ccccc1O")  # phenol
    af = AtomFeaturizer(v2=True)

    for atom in mol.GetAtoms():
        feats = af(atom)
        assert feats.shape[0] == len(af)
        assert torch.all(feats >= 0)
        assert feats.sum() > 0  # at least some one-hots or mass


def test_atom_featurizer_mass_and_aromaticity():
    mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
    af = AtomFeaturizer(v2=True)
    atom = mol.GetAtomWithIdx(0)
    feats = af(atom)
    assert feats[-2] == 1.0  # aromatic flag
    assert torch.isclose(feats[-1], torch.tensor(atom.GetMass() / 100.0))


def test_atom_featurizer_unknown_element():
    mol = Chem.MolFromSmiles("[Xe]")
    af = AtomFeaturizer(v2=True)
    atom = mol.GetAtomWithIdx(0)
    feats = af(atom)
    offset = len(af.atomic_nums)  # +1 slot for unknown is already in __len__
    assert feats[offset] == 1.0  # unknown atomic number


def test_bond_featurizer_all_types():
    mol = Chem.MolFromSmiles("C#CCO")  # single, double, triple
    bf = BondFeaturizer(v2=True)

    for bond in mol.GetBonds():
        feats = bf(bond)
        assert feats.shape[0] == len(bf)
        assert feats.sum() > 0


def test_bond_featurizer_aromatic_conjugated_ring():
    mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
    bf = BondFeaturizer(v2=True)
    bond = mol.GetBondWithIdx(0)
    feats = bf(bond)
    assert feats[0] == 0.0  # bond exists
    conjugation_index = 1 + bf.size_bond_types
    assert feats[conjugation_index] == 1.0  # conjugated
    assert feats[bf.size_bond_types + 1] == 1.0  # in ring


def test_bond_featurizer_none_bond():
    bf = BondFeaturizer(v2=True)
    feats = bf(None)
    assert feats[0] == 1.0  # null flag
    assert feats.sum() == 1.0  # only null flag is on


def test_bond_featurizer_v1_vs_v2_shapes():
    mol = Chem.MolFromSmiles("CC")
    b = mol.GetBondWithIdx(0)

    feats_v1 = BondFeaturizer(v2=False)(b)
    feats_v2 = BondFeaturizer(v2=True)(b)

    assert len(feats_v2) == 15
    assert len(feats_v1) == 14
