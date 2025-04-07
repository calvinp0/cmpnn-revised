import numpy as np
import pytest
from rdkit import Chem

from cmpnn.featurizer.global_feat import (
    CompositeGlobalFeaturizer,
    MorganBinaryFeaturizer,
    RDKit2DFeaturizer,
    RDKit2DNormalizedFeaturizer,
    ChargeFeaturizer
)


@pytest.fixture
def sample_smiles():
    return "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin


def test_morgan_binary_shape(sample_smiles):
    mol = Chem.MolFromSmiles(sample_smiles)
    f = MorganBinaryFeaturizer(radius=2, length=2048, useCountSimulation=True)
    out = f.featurize(mol)
    assert out.shape == (2048,)
    assert set(np.unique(out)).issubset({0, 1})


def test_rdkit2d_shape(sample_smiles):
    mol = Chem.MolFromSmiles(sample_smiles)
    f = RDKit2DFeaturizer()
    out = f.featurize(mol)
    assert out.ndim == 1
    assert out.shape[0] > 0
    assert not np.any(np.isnan(out))


def test_rdkit2d_normalized_shape(sample_smiles):
    mol = Chem.MolFromSmiles(sample_smiles)
    f = RDKit2DNormalizedFeaturizer()
    out = f.featurize(mol)
    assert out.ndim == 1
    assert out.shape[0] > 0
    assert not np.any(np.isnan(out))


def test_charge_featurizer(sample_smiles):
    mol = Chem.MolFromSmiles(sample_smiles)
    f = ChargeFeaturizer()
    out = f(mol)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1,)
    assert out[0] == 0.0  # Aspirin is neutral


def test_composite_featurizer_output_shape(sample_smiles):
    f1 = MorganBinaryFeaturizer(length=128)
    f2 = RDKit2DFeaturizer()
    f3 = ChargeFeaturizer()
    composite = CompositeGlobalFeaturizer([f1, f2, f3])
    out = composite(sample_smiles)
    expected_len = len(f1.featurize(Chem.MolFromSmiles(sample_smiles))) + \
                   len(f2.featurize(Chem.MolFromSmiles(sample_smiles))) + \
                   len(f3(Chem.MolFromSmiles(sample_smiles)))
    assert out.shape == (expected_len,)


def test_invalid_smiles_raises():
    composite = CompositeGlobalFeaturizer([MorganBinaryFeaturizer()])
    with pytest.raises(ValueError):
        composite("invalid_smiles")
